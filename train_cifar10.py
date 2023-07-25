import numpy as np
import torch
from torch.utils import data
from torch import nn
from torch import optim
from torch.cuda import amp
from torchvision import transforms
from torchvision.models import resnet18, resnet50
from models.resnet_cifar10 import resnet20, resnet56
from tqdm import tqdm, trange


BATCH_SIZE = 128*64
MAX_EPOCH = 200
USE_AMP = False
CHANNELS_LAST = False
USE_TF32 = True
# net = resnet20
net = resnet56
# devices = [0]
devices = [0,1]

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


torch.backends.cuda.matmul.allow_tf32 = USE_TF32
torch.backends.cudnn.allow_tf32 = USE_TF32
torch.backends.cudnn.benchmark = True


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


class ToTensor(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1))
        # self.register_buffer('std', torch.tensor(std).view(-1, 1, 1))

    def forward(self, x: torch.Tensor):
        x = x.float()
        x.div_(255.0)
        # x.sub_(self.mean).div_(self.std)
        return x


class DummyCIFAR10(data.Dataset):
    def __init__(self):
        super().__init__()
        self.size = 4096
        self.input = torch.randint(0, 255, (self.size, 3, 32, 32), dtype=torch.uint8)
        self.target = torch.randint(0, 10, size=(self.size, ))

    def __getitem__(self, idx):
        i = idx % self.size
        x = self.input[i]
        y = self.target[i]
        return (x, y)

    def __len__(self):
        return 50000


def icycle(iterable):
    while True:
        for x in iterable:
            yield x


def main():
    dataloader = MultiEpochsDataLoader(
        DummyCIFAR10(),
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )
    dataloader_iter = icycle(dataloader)

    model = nn.Sequential(
        ToTensor(CIFAR10_MEAN, CIFAR10_STD),
        net()
    )

    model = model.to(device='cuda')
    if CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    model = nn.DataParallel(model, device_ids=devices)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_scaler = amp.GradScaler(init_scale=2**14, enabled=USE_AMP)
    criterion = nn.CrossEntropyLoss(reduction='none')

    max_iter = len(dataloader) * MAX_EPOCH
    pbar = trange(max_iter)
    model.train()
    losses = []
    for iter_idx in pbar:
        input, target = next(dataloader_iter)
        if CHANNELS_LAST:
            input = input.to(memory_format=torch.channels_last, non_blocking=True)
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad()
        with amp.autocast(enabled=USE_AMP):
            output = model(input)
            loss = criterion(output, target)
        loss_scaler.scale(loss.mean()).backward()
        if iter_idx < 1:
            loss_scaler.step(optimizer)
        else:
            loss_scaler.unscale_(optimizer)
            optimizer.step()
        loss_scaler.update()
        losses.append(loss.detach().flatten())

        if iter_idx>0:
            pbar.set_description(f"{BATCH_SIZE*pbar.format_dict['rate']:.2f} samples/sec")

        if iter_idx%100 == 0:
            loss = torch.cat(losses).mean()
            losses.clear()
            tqdm.write(f'Train loss: {loss.item():.4f}')


if __name__ == '__main__':
    main()
