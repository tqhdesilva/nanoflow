import json
import os
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, DistributedSampler, SubsetRandomSampler

import config as _config  # noqa: F401
from datasets import ImageNetLatentDataset


def _write_cache(root, train_counts=(3, 2), val_counts=(1,)):
    root = os.fspath(root)
    metadata = {
        "cache_version": 1,
        "source_root": "/source",
        "vae": "stabilityai/sd-vae-ft-ema",
        "transform": {"image_size": 256, "crop": "resize"},
        "latent": {"shape": [4, 32, 32], "dtype": "float16"},
        "splits": {},
    }
    for split, counts in [("train", train_counts), ("val", val_counts)]:
        shards = []
        offset = 0
        os.makedirs(os.path.join(root, split), exist_ok=True)
        for shard_id, count in enumerate(counts):
            rel = f"{split}/shard-{shard_id:05d}.pt"
            values = torch.arange(offset, offset + count, dtype=torch.float16)
            latents = values.view(count, 1, 1, 1).expand(count, 4, 32, 32).clone()
            labels = torch.arange(offset, offset + count, dtype=torch.long) % 1000
            paths = [f"{split}/n00000000_{i:08d}.JPEG" for i in range(offset, offset + count)]
            torch.save(
                {"latents": latents, "labels": labels, "source_paths": paths},
                os.path.join(root, rel),
            )
            shards.append({"file": rel, "count": count})
            offset += count
        metadata["splits"][split] = {
            "count": sum(counts),
            "source_manifest_hash": f"{split}-hash",
            "shards": shards,
        }
    with open(os.path.join(root, "metadata.json"), "w") as f:
        json.dump(metadata, f)


class ImageNetLatentDatasetTest(unittest.TestCase):
    def test_global_index_mapping_and_repeat_reads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_cache(tmpdir)
            ds = ImageNetLatentDataset(cache_root=tmpdir, train=True, lru_cache_size=1)
            x0, y0 = ds[0]
            x2, y2 = ds[2]
            x3, y3 = ds[3]
            x3_again, y3_again = ds[3]

        self.assertEqual(len(ds), 5)
        self.assertEqual(tuple(x0.shape), (4, 32, 32))
        self.assertEqual(x0.dtype, torch.float32)
        self.assertEqual(y0.item(), 0)
        self.assertEqual(float(x2[0, 0, 0]), 2.0)
        self.assertEqual(y2.item(), 2)
        self.assertEqual(float(x3[0, 0, 0]), 3.0)
        self.assertEqual(y3.item(), 3)
        torch.testing.assert_close(x3, x3_again)
        self.assertEqual(y3.item(), y3_again.item())
        self.assertLessEqual(len(ds._shard_cache), 1)

    def test_dataloader_batch_can_span_shards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_cache(tmpdir)
            ds = ImageNetLatentDataset(cache_root=tmpdir, train=True)
            loader = DataLoader(
                ds,
                batch_size=2,
                sampler=SubsetRandomSampler([2, 3]),
                num_workers=0,
            )
            x, y = next(iter(loader))

        self.assertEqual(tuple(x.shape), (2, 4, 32, 32))
        self.assertEqual(sorted(y.tolist()), [2, 3])

    def test_distributed_sampler_partitions_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_cache(tmpdir, train_counts=(4, 4), val_counts=(1,))
            ds = ImageNetLatentDataset(cache_root=tmpdir, train=True)
            shards = [
                list(
                    DistributedSampler(
                        ds,
                        num_replicas=2,
                        rank=rank,
                        shuffle=True,
                        drop_last=True,
                    )
                )
                for rank in range(2)
            ]

        flat = [idx for shard in shards for idx in shard]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(set(flat), set(range(8)))

    def test_metadata_validation_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_cache(tmpdir)
            with self.assertRaises(ValueError):
                ImageNetLatentDataset(cache_root=tmpdir, latent_shape=[8, 16, 16])
            with self.assertRaises(ValueError):
                ImageNetLatentDataset(cache_root=tmpdir, vae="other-vae")

    def test_hydra_latent_dataset_config_materializes(self):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        config_dir = os.path.abspath("configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config", overrides=["dataset=imagenet256_latent"])

        self.assertEqual(cfg.dataset._target_, "datasets.ImageNetLatentDataset")
        self.assertEqual(cfg.dataset.name, "imagenet256_latent")
        self.assertEqual(cfg.dataset.latent_shape, [4, 32, 32])
        self.assertEqual(cfg.dataset.num_classes, 1000)

    def test_hydra_latent_imagenet_experiment_config_materializes(self):
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        config_dir = os.path.abspath("configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=["experiment=imagenet256_latent_cfg"],
            )

        self.assertEqual(cfg.dataset.name, "imagenet256_latent")
        self.assertEqual(cfg.model._target_, "models.ClassCondUNet")
        self.assertEqual(cfg.model.in_ch, 4)
        self.assertEqual(cfg.model.num_classes, 1000)
        self.assertEqual(cfg.training.p_uncond, 0.1)
        self.assertEqual(cfg.sample_logger.latent_shape, [4, 32, 32])
        self.assertEqual(cfg.inference.sampler.latent_shape, [4, 32, 32])
        self.assertEqual(cfg.inference.class_sampler.num_classes, 1000)
        self.assertEqual(cfg.vae.model_id, "stabilityai/sd-vae-ft-ema")


if __name__ == "__main__":
    unittest.main()
