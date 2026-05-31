#!/usr/bin/env python
"""Convert sharded ImageNet latent .pt cache files to mmap friendly .npy files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_keys(obj, keys, context):
    missing = set(keys) - set(obj)
    if missing:
        raise ValueError(f"{context} missing keys: {sorted(missing)}")


def _load_shard(path: Path):
    return torch.load(path, map_location="cpu")


def convert_split(
    input_root: Path, output_root: Path, source_meta, split: str, label_dtype: str
):
    if split not in source_meta.get("splits", {}):
        raise ValueError(f"Source metadata has no split {split!r}")
    split_meta = source_meta["splits"][split]
    _require_keys(split_meta, {"count", "shards"}, f"split {split}")
    count = int(split_meta["count"])
    latent_shape = tuple(source_meta["latent"]["shape"])
    latent_dtype = np.dtype(source_meta["latent"]["dtype"])
    label_dtype_np = np.dtype(label_dtype)

    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    latents_path = split_dir / "latents.npy"
    labels_path = split_dir / "labels.npy"
    source_paths_path = split_dir / "source_paths.txt"

    latents = np.lib.format.open_memmap(
        latents_path,
        mode="w+",
        dtype=latent_dtype,
        shape=(count, *latent_shape),
    )
    labels = np.lib.format.open_memmap(
        labels_path,
        mode="w+",
        dtype=label_dtype_np,
        shape=(count,),
    )

    offset = 0
    with source_paths_path.open("w") as source_paths_file:
        for shard_info in tqdm(split_meta["shards"], desc=split):
            _require_keys(shard_info, {"file", "count"}, f"split {split} shard")
            shard_path = input_root / shard_info["file"]
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing source shard: {shard_path}")
            shard = _load_shard(shard_path)
            _require_keys(shard, {"latents", "labels", "source_paths"}, str(shard_path))
            shard_count = int(shard_info["count"])
            shard_latents = shard["latents"]
            shard_labels = shard["labels"]
            shard_paths = list(shard["source_paths"])
            if shard_latents.shape != (shard_count, *latent_shape):
                raise ValueError(
                    f"Shard {shard_info['file']} latents shape mismatch: {tuple(shard_latents.shape)}"
                )
            if shard_labels.shape[0] != shard_count:
                raise ValueError(f"Shard {shard_info['file']} label count mismatch")
            if len(shard_paths) != shard_count:
                raise ValueError(
                    f"Shard {shard_info['file']} source path count mismatch"
                )
            end = offset + shard_count
            if end > count:
                raise ValueError(f"Split {split} shard counts exceed metadata count")
            latents[offset:end] = shard_latents.numpy().astype(latent_dtype, copy=False)
            labels[offset:end] = shard_labels.numpy().astype(label_dtype_np, copy=False)
            for path in shard_paths:
                source_paths_file.write(f"{path}\n")
            offset = end

    if offset != count:
        raise ValueError(f"Split {split} wrote {offset} rows, expected {count}")
    latents.flush()
    labels.flush()
    return {
        "count": count,
        "source_manifest_hash": split_meta.get("source_manifest_hash"),
        "files": {
            "latents": f"{split}/latents.npy",
            "labels": f"{split}/labels.npy",
            "source_paths": f"{split}/source_paths.txt",
        },
    }


def convert_cache(
    input_root, output_root, splits=("train", "val"), label_dtype="int64"
):
    input_root = Path(input_root)
    output_root = Path(output_root)
    source_metadata_path = input_root / "metadata.json"
    if not source_metadata_path.exists():
        raise FileNotFoundError(f"Missing source metadata: {source_metadata_path}")
    source_meta = json.loads(source_metadata_path.read_text())
    _require_keys(
        source_meta,
        {"cache_version", "vae", "transform", "latent", "splits"},
        "metadata",
    )
    _require_keys(source_meta["latent"], {"shape", "dtype"}, "metadata latent")
    output_root.mkdir(parents=True, exist_ok=True)

    output_meta = {
        "cache_version": int(source_meta["cache_version"]),
        "storage_format": "mmap_npy_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_cache_root": os.path.realpath(input_root),
        "source_metadata_hash": file_sha256(source_metadata_path),
        "vae": source_meta["vae"],
        "vae_torch_dtype": source_meta.get("vae_torch_dtype"),
        "compiled_vae": source_meta.get("compiled_vae"),
        "transform": source_meta["transform"],
        "latent": source_meta["latent"],
        "label": {"dtype": label_dtype},
        "splits": {},
    }
    for split in splits:
        output_meta["splits"][split] = convert_split(
            input_root, output_root, source_meta, split, label_dtype
        )
    (output_root / "metadata.json").write_text(json.dumps(output_meta, indent=2) + "\n")
    return output_meta


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--label-dtype", default="int64")
    args = p.parse_args(argv)
    convert_cache(args.input_root, args.output_root, args.splits, args.label_dtype)


if __name__ == "__main__":
    main()
