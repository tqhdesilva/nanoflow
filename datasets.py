"""Dataset classes: moons, FashionMNIST, CIFAR-10.

Each dataset returns (data, label) tuples and exposes a `num_classes` attribute.
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from sklearn.datasets import make_moons


def _scale_to_minus1_1(x):
    return x * 2 - 1


class MoonsDataset(Dataset):
    num_classes = 2

    def __init__(self, n=8000, noise=0.05, train=True, **kwargs):
        X, y = make_moons(n_samples=n, noise=noise)
        X = np.asarray(X)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        split = int(0.8 * len(X))
        if train:
            self.data, self.labels = X[:split], y[:split]
        else:
            self.data, self.labels = X[split:], y[split:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class FashionMNISTDataset(Dataset):
    num_classes = 10

    def __init__(self, root="./data", train=True, **kwargs):
        from torchvision import datasets, transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(_scale_to_minus1_1),
            ]
        )
        self._ds = datasets.FashionMNIST(
            root=root, train=train, download=True, transform=transform
        )

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]  # (image, label)


class CifarDataset(Dataset):
    num_classes = 10

    def __init__(self, root="./data", train=True, **kwargs):
        from torchvision import datasets, transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(_scale_to_minus1_1),
            ]
        )
        self._ds = datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform
        )

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]  # (image, label)


def build_dataloader(dataset, batch_size, num_workers, train, **_):
    """Build a (possibly DDP-distributed) DataLoader.

    `dataset` is a Hydra partial called here with `train=...`. When WORLD_SIZE>1
    (set by torchrun), wrap in a DistributedSampler so each rank sees a disjoint shard.
    """
    ds = dataset(train=train)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    sampler = None
    shuffle = train
    if world_size > 1:
        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=train
        )
        shuffle = False
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )
