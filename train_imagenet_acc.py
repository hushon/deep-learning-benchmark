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
from accelerate import Accelerator

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True

BATCH_SIZE = 64*8
MAX_ITER = 100 * 1281167 // BATCH_SIZE
USE_AMP = True 
# net = resnet18
net = resnet50
# devices = [0]
# devices = [0,1]
devices = [0,1,2,3]
# devices = [0,1,2,3,4,5,6,7]


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


def icycle(iterable):
    while True:
        for x in iterable:
            yield x


class DummyDataset(data.Dataset):
    def __init__(self):
        super().__init__()
        self.size = 10000
        self.input = torch.randint(0, 255, (self.size, 3, 224, 224), dtype=torch.uint8)
        self.target = torch.randint(0, 1000, size=(self.size, ))

    def __getitem__(self, idx):
        x = self.input[idx]
        y = self.target[idx]
        return (x, y)

    def __len__(self):
        return self.size


class ToTensor(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(-1, 1, 1))

    def forward(self, x: torch.Tensor):
        x = x.float()
        x.div_(255.0)
        x.sub_(self.mean).div_(self.std)
        return x


def main():
    accelerator = Accelerator(fp16=USE_AMP)

    dataloader = MultiEpochsDataLoader(
        DummyDataset(),
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )

    model = nn.Sequential(
        ToTensor(IMAGENET_MEAN, IMAGENET_STD),
        net(),
    ).to(memory_format=torch.channels_last)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    dataloader_iter = icycle(dataloader)

#    input = torch.randint(0, 255, (BATCH_SIZE, 3, 224, 224), dtype=torch.uint8).pin_memory()
#    input = torch.empty((BATCH_SIZE, 3, 224, 224), dtype=torch.uint8, pin_memory=True)
    # target_orig = torch.randint(0, 1000, size=(BATCH_SIZE, )).pin_memory()

    pbar = trange(MAX_ITER)
    model.train()
    for iter_idx in pbar:
        input, target = next(dataloader_iter)
        output = model(input)
        loss = criterion(output, target)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        if iter_idx>0:
            pbar.set_description(f"{BATCH_SIZE*pbar.format_dict['rate']:.2f} samples/sec")


if __name__ == '__main__':
    main()
