import torch
from torch import nn, utils, optim, cuda
from torch.utils import data
from torch.cuda import amp
import tqdm

from torchvision.models import resnet18, resnet34,resnet50
from torchvision import datasets
from torch.nn.parallel import DistributedDataParallel
import torch.utils.data
import torch.distributed
import torch.multiprocessing

import numpy as np
# import albumentations as A
from PIL import Image
from simplejpeg import decode_jpeg
import os


torch.backends.cudnn.benchmark = True

N_GPUS = torch.cuda.device_count()
BATCH_SIZE = 64*32
MAX_EPOCH = 100
USE_AMP = True
net = resnet18
# net = resnet34
# net = resnet50
DATA_ROOT = '/ramdisk/'


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MultiEpochsDataLoader(data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def cycle(
        data_loader: data.DataLoader,
        distributed_sampler: data.distributed.DistributedSampler,
        ):
    epoch = 0
    while True:
        epoch += 1
        distributed_sampler.set_epoch(epoch)
        for x in data_loader:
            yield x


def image_loader(path: str):
    try:
        with open(path, 'rb') as fp:
            image = decode_jpeg(fp.read(), colorspace='RGB')
    except:
        image = Image.open(path).convert('RGB')
        image = np.asarray(image)
    return image



class DummyImagenet(data.Dataset):
    def __init__(self):
        super().__init__()
        self.size = 1024
        self.input = torch.randint(0, 255, (self.size, 3, 224, 224), dtype=torch.uint8, generator=torch.Generator().manual_seed(42))
        self.target = torch.randint(0, 1000, size=(self.size, ), generator=torch.Generator().manual_seed(42))

    def __getitem__(self, idx):
        i = idx % self.size
        x = self.input[i]
        y = self.target[i]
        return (x, y)

    def __len__(self):
        return 1281167


class ToTensor(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor):
        x = x.float()
        x.div_(255.0)
        x.sub_(self.mean).div_(self.std)
        return x


class ImageFolder(datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(image=sample)['image']
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


def main(rank, world_size):
    torch.distributed.init_process_group(
                                        backend=torch.distributed.Backend.NCCL,
                                        init_method='tcp://localhost:12345',
                                        rank=rank,
                                        world_size=world_size
                                        )

    # transform_train_a = A.Compose([
    #     A.RandomResizedCrop(224, 224),
    #     A.HorizontalFlip(),
    #     ])
    train_dataset = DummyImagenet()
    # train_dataset = ImageFolder(os.path.join(DATA_ROOT, 'imagenet', 'train'), transform=transform_train_a, loader=image_loader)
    train_sampler = data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_dataloader = MultiEpochsDataLoader(
                                    train_dataset,
                                    batch_size=BATCH_SIZE//world_size,
                                    sampler=train_sampler,
                                    num_workers=4,
                                    pin_memory=False
                                    )
    train_dataloader_iter = cycle(train_dataloader, train_sampler)


    model = nn.Sequential(
        ToTensor(IMAGENET_MEAN, IMAGENET_STD),
        net(),
    )


    model = model.to(device=rank)
    if USE_AMP: model = model.to(memory_format=torch.channels_last)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    loss_scaler = amp.GradScaler(init_scale=2**14, enabled=USE_AMP)

    model.train()
    max_iter = len(train_dataloader) * MAX_EPOCH
    progress_bar = tqdm.trange(max_iter, disable=(rank!=0))
    losses = []

    for iter_idx in progress_bar:
        input, target = next(train_dataloader_iter)

        input = input.to(device=rank, non_blocking=True).permute(0, 3, 1, 2)
        if USE_AMP: input = input.to(memory_format=torch.channels_last)
        target = target.to(device=rank, non_blocking=True)

        optimizer.zero_grad()
        with amp.autocast(enabled=USE_AMP):
            output = model(input)
            loss = criterion(output, target)
        loss_scaler.scale(loss.mean()).backward()
        # if iter_idx < 1:
        #     loss_scaler.step(optimizer)
        # else:
        #     loss_scaler.unscale_(optimizer)
        #     optimizer.step()
        loss_scaler.step(optimizer)
        loss_scaler.update()
        losses.append(loss.detach().flatten())

        if iter_idx>0 and rank==0:
            progress_bar.set_description(f"{BATCH_SIZE*progress_bar.format_dict['rate']:.2f} samples/sec")

        if iter_idx%100 == 0:
            loss = torch.cat(losses).mean()
            torch.distributed.reduce(loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            losses.clear()
            if rank==0:
                tqdm.tqdm.write(f'Train loss: {loss.item()/world_size:.4f}')


    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    torch.multiprocessing.spawn(
                                main,
                                args=(N_GPUS,),
                                nprocs=N_GPUS,
                                join=True
                                )
