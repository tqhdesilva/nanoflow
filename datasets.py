"""Dataset loaders — moons, FashionMNIST, CIFAR-10."""

import torch
from sklearn.datasets import make_moons


def moons_dataset(n=8000, noise=0.05):
    """2D moons dataset, normalized to roughly [-1, 1]."""
    X, _ = make_moons(n_samples=n, noise=noise)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return torch.tensor(X, dtype=torch.float32)


def fashion_dataset(root="./data"):
    """FashionMNIST, scaled to [-1, 1]. Returns (60000, 1, 28, 28) tensor."""
    from torchvision import datasets, transforms
    ds = datasets.FashionMNIST(
        root=root, train=True, download=True,
        transform=transforms.ToTensor(),
    )
    X = torch.stack([img for img, _ in ds])  # (60000, 1, 28, 28) in [0, 1]
    return X * 2 - 1


def cifar_dataset(root="./data"):
    """CIFAR-10, scaled to [-1, 1]. Returns (50000, 3, 32, 32) tensor."""
    import torchvision
    ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    X = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float() / 255.0  # (50000, 3, 32, 32)
    return X * 2 - 1
