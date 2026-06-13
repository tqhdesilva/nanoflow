#!/usr/bin/env python3
"""Preflight checks for the ImageNet-256 latent mmap cache."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import ImageNetLatentMMapDataset, build_dataloader

EXPECTED_LATENT_SHAPE = (4, 32, 32)
EXPECTED_LATENT_DTYPE = np.dtype("float16")
EXPECTED_LABEL_DTYPE = np.dtype("int64")
NUM_CLASSES = 1000


def preflight_cache(
    cache_root: str | Path,
    *,
    batch_size: int = 2,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Validate an ImageNet-256 latent mmap cache and load smoke batches.

    Args:
        cache_root: Directory containing `metadata.json`, `train`, and `val`.
        batch_size: Batch size for the train and val dataloader smoke reads.
        num_workers: Number of dataloader workers for smoke reads.

    Returns:
        JSON-compatible summary of split arrays and loaded batches.
    """
    root = Path(cache_root)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")

    metadata = _load_metadata(root)
    summary: dict[str, Any] = {
        "cache_root": str(root),
        "metadata": {
            "storage_format": metadata.get("storage_format"),
            "cache_version": metadata.get("cache_version"),
            "vae": metadata.get("vae"),
        },
        "splits": {},
        "batches": {},
    }
    for split in ("train", "val"):
        summary["splits"][split] = _validate_split(root, metadata, split)

    dataset_factory = partial(ImageNetLatentMMapDataset, cache_root=str(root))
    for split, train in (("train", True), ("val", False)):
        loader = build_dataloader(
            dataset_factory,
            batch_size=batch_size,
            num_workers=num_workers,
            train=train,
            pin_memory=False,
            persistent_workers=False,
        )
        latents, labels = next(iter(loader))
        _validate_batch(split, latents, labels)
        summary["batches"][split] = {
            "latents_shape": list(latents.shape),
            "latents_dtype": str(latents.dtype),
            "labels_shape": list(labels.shape),
            "labels_dtype": str(labels.dtype),
            "label_min": int(labels.min().item()),
            "label_max": int(labels.max().item()),
        }
    return summary


def _load_metadata(root: Path) -> dict[str, Any]:
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing metadata.json at {metadata_path}")
    with open(metadata_path) as handle:
        metadata = json.load(handle)
    _require_equal("storage_format", metadata.get("storage_format"), "mmap_npy_v1")
    _require_equal("cache_version", int(metadata.get("cache_version", -1)), 1)
    _require_equal("vae", metadata.get("vae"), "stabilityai/sd-vae-ft-ema")
    _require_equal(
        "latent.shape",
        tuple(metadata.get("latent", {}).get("shape", [])),
        EXPECTED_LATENT_SHAPE,
    )
    _require_equal(
        "latent.dtype",
        np.dtype(metadata.get("latent", {}).get("dtype", "")),
        EXPECTED_LATENT_DTYPE,
    )
    _require_equal(
        "label.dtype",
        np.dtype(metadata.get("label", {}).get("dtype", "")),
        EXPECTED_LABEL_DTYPE,
    )
    _require_equal(
        "transform.image_size",
        int(metadata.get("transform", {}).get("image_size", -1)),
        256,
    )
    _require_equal(
        "transform.crop", metadata.get("transform", {}).get("crop"), "resize"
    )
    for split in ("train", "val"):
        if split not in metadata.get("splits", {}):
            raise ValueError(f"metadata missing split {split!r}")
    return metadata


def _validate_split(root: Path, metadata: dict[str, Any], split: str) -> dict[str, Any]:
    split_meta = metadata["splits"][split]
    count = int(split_meta.get("count", -1))
    if count <= 0:
        raise ValueError(f"split {split!r} count must be positive")
    files = split_meta.get("files", {})
    latents_path = root / files.get("latents", "")
    labels_path = root / files.get("labels", "")
    source_paths_path = root / files.get("source_paths", "")
    for path in (latents_path, labels_path, source_paths_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing mmap cache file: {path}")

    latents = np.load(latents_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    expected_latents_shape = (count, *EXPECTED_LATENT_SHAPE)
    _require_equal(
        f"{split}.latents.shape", tuple(latents.shape), expected_latents_shape
    )
    _require_equal(f"{split}.latents.dtype", latents.dtype, EXPECTED_LATENT_DTYPE)
    _require_equal(f"{split}.labels.shape", tuple(labels.shape), (count,))
    _require_equal(f"{split}.labels.dtype", labels.dtype, EXPECTED_LABEL_DTYPE)
    source_path_count = _count_lines(source_paths_path)
    _require_equal(f"{split}.source_paths.count", source_path_count, count)
    label_min = int(labels.min())
    label_max = int(labels.max())
    if label_min < 0 or label_max >= NUM_CLASSES:
        raise ValueError(
            f"{split} label range must be within [0, {NUM_CLASSES - 1}], "
            f"got [{label_min}, {label_max}]"
        )
    return {
        "count": count,
        "latents_shape": list(latents.shape),
        "latents_dtype": str(latents.dtype),
        "labels_shape": list(labels.shape),
        "labels_dtype": str(labels.dtype),
        "label_min": label_min,
        "label_max": label_max,
        "source_paths_count": source_path_count,
    }


def _count_lines(path: Path) -> int:
    with open(path) as handle:
        return sum(1 for _ in handle)


def _validate_batch(split: str, latents: torch.Tensor, labels: torch.Tensor) -> None:
    if tuple(latents.shape[1:]) != EXPECTED_LATENT_SHAPE:
        raise ValueError(
            f"{split} batch latent shape must end with {EXPECTED_LATENT_SHAPE}, "
            f"got {tuple(latents.shape)}"
        )
    if latents.dtype != torch.float32:
        raise ValueError(f"{split} batch latents must be float32, got {latents.dtype}")
    if labels.dtype != torch.long:
        raise ValueError(f"{split} batch labels must be int64, got {labels.dtype}")
    label_min = int(labels.min().item())
    label_max = int(labels.max().item())
    if label_min < 0 or label_max >= NUM_CLASSES:
        raise ValueError(
            f"{split} batch label range must be within [0, {NUM_CLASSES - 1}], "
            f"got [{label_min}, {label_max}]"
        )


def _require_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"expected {name}={expected!r}, got {actual!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, help="latent mmap cache root")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Run the preflight from the command line."""
    args = _parse_args()
    summary = preflight_cache(
        args.cache_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
