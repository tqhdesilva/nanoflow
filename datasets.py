"""Dataset classes: moons, FashionMNIST, CIFAR-10, and ImageNet 256.

Each dataset returns (data, label) tuples and exposes a `num_classes` attribute.
"""

import fcntl
import os

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from sklearn.datasets import make_moons


def _scale_to_minus1_1(x):
    return x * 2 - 1


def _convert_to_rgb(image):
    return image.convert("RGB")


class FileLockMixin:
    """Serialize constructor-time dataset preparation with a POSIX file lock."""

    lock_filename = ".nanoflow_imagenet_prepare.lock"

    @classmethod
    def _default_lock_path(cls, root):
        return os.path.join(os.path.realpath(os.fspath(root)), cls.lock_filename)

    @classmethod
    def _resolve_lock_path(cls, root, lock_path=None):
        if lock_path is None:
            return cls._default_lock_path(root)
        return os.path.realpath(os.fspath(lock_path))

    def __init__(self, root, *args, lock_path=None, **kwargs):
        resolved_lock_path = self._resolve_lock_path(root, lock_path)
        lock_dir = os.path.dirname(resolved_lock_path)
        if lock_dir:
            os.makedirs(lock_dir, exist_ok=True)
        fd = os.open(resolved_lock_path, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            self.prepare_lock_path = resolved_lock_path
            super().__init__(root, *args, **kwargs)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


class DistributedImageNet(FileLockMixin, torchvision.datasets.ImageNet):
    """ImageNet wrapper that serializes torchvision root preparation per root."""


class MoonsDataset(Dataset):
    num_classes = 2

    def __init__(self, n=8000, noise=0.05, train=True, name="moons"):
        self.name = name
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

    def __init__(self, root="./data", train=True, name="fashion"):
        from torchvision import datasets, transforms

        self.name = name
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

    def __init__(self, root="./data", train=True, name="cifar10"):
        from torchvision import datasets, transforms

        self.name = name
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


class ImageNet256Dataset(Dataset):
    num_classes = 1000

    def __init__(
        self,
        root="data/imagenet",
        train=True,
        name="imagenet256",
        image_size=256,
        train_crop="random_resized",
        val_crop="center",
        hflip=True,
        lock_path=None,
        num_classes=1000,
    ):
        if num_classes != self.num_classes:
            raise ValueError(
                f"ImageNet256Dataset num_classes must be {self.num_classes}"
            )
        self.name = name
        self.root = root
        self.train = train
        self.image_size = image_size
        self.train_crop = train_crop
        self.val_crop = val_crop
        self.hflip = hflip
        self.lock_path = lock_path
        transform = self._build_transform(
            train=train,
            image_size=image_size,
            train_crop=train_crop,
            val_crop=val_crop,
            hflip=hflip,
        )
        split = "train" if train else "val"
        self._ds = DistributedImageNet(
            root=root,
            split=split,
            transform=transform,
            lock_path=lock_path,
        )

    @staticmethod
    def _resize_size(image_size):
        return int(round(image_size * 256 / 224))

    @classmethod
    def _build_transform(cls, train, image_size, train_crop, val_crop, hflip):
        from torchvision import transforms

        ops = [transforms.Lambda(_convert_to_rgb)]
        resize_size = cls._resize_size(image_size)
        if train:
            if train_crop == "random_resized":
                ops.append(transforms.RandomResizedCrop(image_size))
            elif train_crop == "center":
                ops.extend(
                    [
                        transforms.Resize(resize_size),
                        transforms.CenterCrop(image_size),
                    ]
                )
            else:
                raise ValueError(f"Unknown train_crop: {train_crop!r}")
            if hflip:
                ops.append(transforms.RandomHorizontalFlip())
        else:
            if val_crop != "center":
                raise ValueError(f"Unknown val_crop: {val_crop!r}")
            ops.extend(
                [
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(image_size),
                ]
            )
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Lambda(_scale_to_minus1_1),
            ]
        )
        return transforms.Compose(ops)

    @property
    def classes(self):
        return self._ds.classes

    @property
    def class_to_idx(self):
        return self._ds.class_to_idx

    @property
    def samples(self):
        return self._ds.samples

    @property
    def targets(self):
        return self._ds.targets

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]


def build_dataloader(
    dataset,
    batch_size,
    num_workers,
    train,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=2,
):
    """Build a DataLoader, using DistributedSampler when WORLD_SIZE is greater than 1."""
    ds = dataset(train=train)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    sampler = None
    shuffle = train
    if world_size > 1:
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=train,
            drop_last=train,
        )
        shuffle = False

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": train,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(ds, **loader_kwargs)
