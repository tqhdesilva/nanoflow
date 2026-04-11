"""Dataset loaders — moons, FashionMNIST, CIFAR-10.

Each function returns a torch.utils.data.Dataset (not a raw tensor).
"""

import torch
from torch.utils.data import Dataset, TensorDataset

from sklearn.datasets import make_moons


def _scale_to_minus1_1(x):
    return x * 2 - 1


def moons_dataset(n=8000, noise=0.05, train=True):
    """2D moons dataset, normalized to roughly [-1, 1]."""
    X, _ = make_moons(n_samples=n, noise=noise)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = torch.tensor(X, dtype=torch.float32)
    split = int(0.8 * len(X))
    return TensorDataset(X[:split]) if train else TensorDataset(X[split:])


def fashion_dataset(root="./data", train=True):
    """FashionMNIST, scaled to [-1, 1]."""
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(_scale_to_minus1_1),
    ])
    return datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)


def cifar_dataset(root="./data", train=True):
    """CIFAR-10, scaled to [-1, 1]."""
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(_scale_to_minus1_1),
    ])
    return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
