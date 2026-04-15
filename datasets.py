"""Dataset classes — moons, FashionMNIST, CIFAR-10.

Each dataset returns (data, label) tuples and exposes a `num_classes` attribute.
"""

import torch
from torch.utils.data import Dataset

from sklearn.datasets import make_moons


def _scale_to_minus1_1(x):
    return x * 2 - 1


class MoonsDataset(Dataset):
    num_classes = 2

    def __init__(self, n=8000, noise=0.05, train=True, **kwargs):
        X, y = make_moons(n_samples=n, noise=noise)
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
