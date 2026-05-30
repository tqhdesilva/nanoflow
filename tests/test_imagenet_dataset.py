import inspect
import multiprocessing as mp
import os
import socket
import tempfile
import time
import traceback
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image
from torch.utils.data import Dataset

import config as _config  # noqa: F401
from datasets import (
    CifarDataset,
    FashionMNISTDataset,
    FileLockMixin,
    ImageNet256Dataset,
    MoonsDataset,
    build_dataloader,
)


class FakeImageNet:
    def __init__(self, root, split="train", transform=None, lock_path=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.lock_path = lock_path
        self.classes = ["n00000000", "n00000001"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = [
            (os.path.join(root, split, self.classes[0], "0.JPEG"), 0),
            (os.path.join(root, split, self.classes[1], "1.JPEG"), 1),
        ]
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx == 0:
            image = Image.new("L", (320, 300), color=128)
        else:
            image = Image.new("RGBA", (300, 320), color=(64, 128, 192, 255))
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[idx]


class TinyImageNet18(FakeImageNet):
    def __init__(self, root, split="train", transform=None, lock_path=None):
        super().__init__(root, split=split, transform=transform, lock_path=lock_path)
        self.samples = [
            (os.path.join(root, split, f"n{i % 2:08d}", f"{i}.JPEG"), i % 2)
            for i in range(18)
        ]
        self.targets = [label for _, label in self.samples]


class TinyDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, idx):
        return torch.tensor([idx]), idx


class RecordingBase:
    def __init__(
        self,
        root,
        active,
        max_active,
        global_active,
        global_max_active,
        sleep_seconds,
    ):
        self.root = root
        with active.get_lock():
            active.value += 1
            current = active.value
        with max_active.get_lock():
            max_active.value = max(max_active.value, current)
        with global_active.get_lock():
            global_active.value += 1
            global_current = global_active.value
        with global_max_active.get_lock():
            global_max_active.value = max(global_max_active.value, global_current)
        time.sleep(sleep_seconds)
        with global_active.get_lock():
            global_active.value -= 1
        with active.get_lock():
            active.value -= 1


class LockedRecordingDataset(FileLockMixin, RecordingBase):
    pass


def _lock_worker(
    root,
    active,
    max_active,
    global_active,
    global_max_active,
    sleep_seconds,
    barrier,
    error_queue,
    lock_path=None,
):
    try:
        barrier.wait(timeout=10)
        LockedRecordingDataset(
            root,
            active,
            max_active,
            global_active,
            global_max_active,
            sleep_seconds,
            lock_path=lock_path,
        )
    except Exception:
        error_queue.put(traceback.format_exc())
        raise


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _ddp_sampler_worker(rank, world_size, port, config_dir, result_queue):
    try:
        os.environ.update(
            {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "WORLD_SIZE": str(world_size),
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
            }
        )
        import hydra
        import torch.distributed as dist
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        import datasets as datasets_module

        datasets_module.DistributedImageNet = TinyImageNet18
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=imagenet256_cfg",
                    "training.batch_size=1",
                    "training.num_workers=0",
                ],
            )
        loader = hydra.utils.instantiate(cfg.train_loader)
        sampler = loader.sampler
        epoch0 = list(iter(sampler))
        sampler.set_epoch(1)
        epoch1 = list(iter(sampler))
        result_queue.put((rank, epoch0, epoch1, type(sampler).__name__))
        dist.destroy_process_group()
    except Exception:
        result_queue.put(("error", rank, traceback.format_exc()))
        raise


class ImageNetDatasetTest(unittest.TestCase):
    def test_dataset_constructors_do_not_accept_unused_kwargs(self):
        for constructor in [
            MoonsDataset.__init__,
            FashionMNISTDataset.__init__,
            CifarDataset.__init__,
            ImageNet256Dataset.__init__,
            build_dataloader,
        ]:
            with self.subTest(constructor=constructor):
                params = inspect.signature(constructor).parameters.values()
                self.assertFalse(any(p.kind == p.VAR_KEYWORD for p in params))

    def test_imagenet_transforms_rgb_and_scale_to_minus1_1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("datasets.DistributedImageNet", FakeImageNet):
                ds = ImageNet256Dataset(
                    root=tmpdir,
                    train=True,
                    image_size=256,
                    train_crop="center",
                    hflip=False,
                )
                x, y = ds[0]
                x_again, _ = ds[0]

        self.assertEqual(ds.num_classes, 1000)
        self.assertEqual(ds.class_to_idx, {"n00000000": 0, "n00000001": 1})
        self.assertEqual(tuple(x.shape), (3, 256, 256))
        self.assertEqual(x.dtype, torch.float32)
        self.assertGreaterEqual(float(x.min()), -1.0)
        self.assertLessEqual(float(x.max()), 1.0)
        self.assertEqual(y, 0)
        torch.testing.assert_close(x[0], x[1])
        torch.testing.assert_close(x, x_again)

    def test_imagenet_validation_transform_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("datasets.DistributedImageNet", FakeImageNet):
                ds = ImageNet256Dataset(root=tmpdir, train=False, image_size=256)
                x1, y1 = ds[1]
                x2, y2 = ds[1]

        self.assertEqual(tuple(x1.shape), (3, 256, 256))
        self.assertEqual(y1, y2)
        torch.testing.assert_close(x1, x2)

    def test_invalid_crop_names_fail_loudly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("datasets.DistributedImageNet", FakeImageNet):
                with self.assertRaises(ValueError):
                    ImageNet256Dataset(root=tmpdir, train=True, train_crop="bad")
                with self.assertRaises(ValueError):
                    ImageNet256Dataset(root=tmpdir, train=False, val_crop="bad")

    def test_dataloader_explicit_runtime_options(self):
        def dataset_factory(train):
            return TinyDataset()

        calls = []

        def fake_loader(dataset, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(dataset=dataset, sampler=kwargs.get("sampler"))

        with mock.patch("datasets.DataLoader", fake_loader):
            with mock.patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0"}):
                build_dataloader(
                    dataset_factory,
                    batch_size=2,
                    num_workers=0,
                    train=True,
                    pin_memory=False,
                    persistent_workers=False,
                    prefetch_factor=7,
                )
                build_dataloader(
                    dataset_factory,
                    batch_size=2,
                    num_workers=2,
                    train=True,
                    pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=7,
                )

        self.assertFalse(calls[0]["pin_memory"])
        self.assertFalse(calls[0]["persistent_workers"])
        self.assertNotIn("prefetch_factor", calls[0])
        self.assertTrue(calls[1]["pin_memory"])
        self.assertTrue(calls[1]["persistent_workers"])
        self.assertEqual(calls[1]["prefetch_factor"], 7)

    def test_hydra_imagenet_dataset_config_materializes(self):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        config_dir = os.path.abspath("configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="config", overrides=["experiment=imagenet256_cfg"]
            )

        self.assertEqual(cfg.dataset._target_, "datasets.ImageNet256Dataset")
        self.assertEqual(cfg.dataset.name, "imagenet256")
        self.assertEqual(cfg.dataset.num_classes, 1000)
        self.assertEqual(cfg.dataset.image_size, 256)
        self.assertEqual(cfg.train_loader.pin_memory, True)
        self.assertEqual(cfg.train_loader.persistent_workers, False)
        self.assertEqual(cfg.train_loader.prefetch_factor, 2)

    def _run_lock_specs(self, specs):
        ctx = mp.get_context("spawn")
        barrier = ctx.Barrier(len(specs))
        error_queue = ctx.Queue()
        processes = []
        for spec in specs:
            process = ctx.Process(
                target=_lock_worker,
                args=(
                    spec["root"],
                    spec["active"],
                    spec["max_active"],
                    spec["global_active"],
                    spec["global_max_active"],
                    spec.get("sleep_seconds", 0.1),
                    barrier,
                    error_queue,
                    spec.get("lock_path"),
                ),
            )
            processes.append(process)
            process.start()
        for process in processes:
            process.join(timeout=15)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
                self.fail("lock worker timed out")
        errors = []
        while not error_queue.empty():
            errors.append(error_queue.get())
        self.assertEqual(errors, [])
        self.assertEqual(
            [process.exitcode for process in processes], [0] * len(processes)
        )

    def test_default_root_lock_serializes_one_root(self):
        ctx = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as root:
            active = ctx.Value("i", 0)
            max_active = ctx.Value("i", 0)
            global_active = ctx.Value("i", 0)
            global_max = ctx.Value("i", 0)
            specs = [
                {
                    "root": root,
                    "active": active,
                    "max_active": max_active,
                    "global_active": global_active,
                    "global_max_active": global_max,
                }
                for _ in range(4)
            ]
            self._run_lock_specs(specs)
            self.assertEqual(max_active.value, 1)
            self.assertEqual(global_max.value, 1)

    def test_default_root_lock_allows_different_roots_to_prepare_concurrently(self):
        ctx = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmpdir:
            root0 = os.path.join(tmpdir, "host0", "imagenet")
            root1 = os.path.join(tmpdir, "host1", "imagenet")
            os.makedirs(root0)
            os.makedirs(root1)
            active0 = ctx.Value("i", 0)
            active1 = ctx.Value("i", 0)
            max0 = ctx.Value("i", 0)
            max1 = ctx.Value("i", 0)
            global_active = ctx.Value("i", 0)
            global_max = ctx.Value("i", 0)
            specs = []
            for root, active, max_active in [
                (root0, active0, max0),
                (root0, active0, max0),
                (root1, active1, max1),
                (root1, active1, max1),
            ]:
                specs.append(
                    {
                        "root": root,
                        "active": active,
                        "max_active": max_active,
                        "global_active": global_active,
                        "global_max_active": global_max,
                        "sleep_seconds": 0.2,
                    }
                )
            self._run_lock_specs(specs)
            self.assertEqual(max0.value, 1)
            self.assertEqual(max1.value, 1)
            self.assertGreater(global_max.value, 1)

    def test_explicit_shared_lock_serializes_different_roots(self):
        ctx = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmpdir:
            root0 = os.path.join(tmpdir, "host0", "imagenet")
            root1 = os.path.join(tmpdir, "host1", "imagenet")
            os.makedirs(root0)
            os.makedirs(root1)
            shared_lock = os.path.join(tmpdir, "shared.lock")
            global_active = ctx.Value("i", 0)
            global_max = ctx.Value("i", 0)
            specs = []
            for root in [root0, root0, root1, root1]:
                specs.append(
                    {
                        "root": root,
                        "active": ctx.Value("i", 0),
                        "max_active": ctx.Value("i", 0),
                        "global_active": global_active,
                        "global_max_active": global_max,
                        "sleep_seconds": 0.1,
                        "lock_path": shared_lock,
                    }
                )
            self._run_lock_specs(specs)
            self.assertEqual(global_max.value, 1)

    def test_default_lock_serializes_real_and_symlinked_root(self):
        ctx = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "imagenet")
            link = os.path.join(tmpdir, "imagenet_link")
            os.makedirs(root)
            try:
                os.symlink(root, link)
            except OSError as exc:
                self.skipTest(f"symlink creation failed: {exc}")
            active = ctx.Value("i", 0)
            max_active = ctx.Value("i", 0)
            global_active = ctx.Value("i", 0)
            global_max = ctx.Value("i", 0)
            specs = [
                {
                    "root": path,
                    "active": active,
                    "max_active": max_active,
                    "global_active": global_active,
                    "global_max_active": global_max,
                    "sleep_seconds": 0.15,
                }
                for path in [root, link]
            ]
            self._run_lock_specs(specs)
            self.assertEqual(max_active.value, 1)
            self.assertEqual(global_max.value, 1)

    def test_fake_per_host_imagenet_membership_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root0 = os.path.join(tmpdir, "host0", "imagenet")
            root1 = os.path.join(tmpdir, "host1", "imagenet")
            os.makedirs(root0)
            os.makedirs(root1)
            with mock.patch("datasets.DistributedImageNet", TinyImageNet18):
                ds0 = ImageNet256Dataset(root=root0, train=True)
                ds1 = ImageNet256Dataset(root=root1, train=True)

            rel0 = [
                (os.path.relpath(path, root0), label) for path, label in ds0.samples
            ]
            rel1 = [
                (os.path.relpath(path, root1), label) for path, label in ds1.samples
            ]
            self.assertEqual(ds0.class_to_idx, ds1.class_to_idx)
            self.assertEqual(rel0, rel1)
            self.assertEqual(ds0.targets, ds1.targets)

    def test_synthetic_ddp_sampler_shards_imagenet_config(self):
        ctx = mp.get_context("spawn")
        world_size = 4
        port = _free_port()
        result_queue = ctx.Queue()
        config_dir = os.path.abspath("configs")
        processes = [
            ctx.Process(
                target=_ddp_sampler_worker,
                args=(rank, world_size, port, config_dir, result_queue),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=20)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
                self.fail("ddp sampler worker timed out")
        results = []
        while not result_queue.empty():
            item = result_queue.get()
            if item[0] == "error":
                self.fail(item[2])
            results.append(item)
        self.assertEqual([process.exitcode for process in processes], [0] * world_size)
        results.sort(key=lambda item: item[0])
        self.assertEqual(
            [item[3] for item in results], ["DistributedSampler"] * world_size
        )

        epoch0 = [item[1] for item in results]
        epoch1 = [item[2] for item in results]
        self.assertEqual([len(shard) for shard in epoch0], [4] * world_size)
        self.assertEqual([len(shard) for shard in epoch1], [4] * world_size)
        flat0 = [idx for shard in epoch0 for idx in shard]
        flat1 = [idx for shard in epoch1 for idx in shard]
        self.assertEqual(len(flat0), len(set(flat0)))
        self.assertEqual(len(flat1), len(set(flat1)))
        self.assertTrue(set(flat0).issubset(set(range(18))))
        self.assertTrue(set(flat1).issubset(set(range(18))))
        self.assertNotEqual(epoch0, epoch1)


if __name__ == "__main__":
    unittest.main()
