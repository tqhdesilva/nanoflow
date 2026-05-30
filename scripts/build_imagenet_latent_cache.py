#!/usr/bin/env python
"""Build sharded latents from the Kaggle ImageNet256 tree."""
from __future__ import annotations

import argparse, hashlib, json, os, re, sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT))
from image_transforms import build_cache_transform
from vae import VAEWrapper

WNID = re.compile(r"n\d{8}")
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_labels(root: Path) -> dict[str, int]:
    data = json.loads((root / "imagenet_class_index.json").read_text())
    return {v[0]: int(k) for k, v in data.items()}


def find_samples(root: Path, split: str, limit: int | None):
    labels = load_labels(root)
    files = sorted(p for p in (root / split).iterdir() if p.suffix.lower() in EXTS)
    out = []
    for path in files[:limit]:
        ids = WNID.findall(path.name)
        if not ids or ids[-1] not in labels:
            raise ValueError(f"Cannot infer label from filename: {path}")
        out.append((path, labels[ids[-1]]))
    return out


def manifest_hash(samples, root: Path) -> str:
    h = hashlib.sha256()
    for path, label in samples:
        rel = path.relative_to(root).as_posix()
        h.update(f"{rel} {label} {path.stat().st_size}\n".encode())
    return h.hexdigest()


def load_batch(root: Path, samples, transform):
    images, labels, paths = [], [], []
    for path, label in samples:
        with Image.open(path) as im:
            images.append(transform(im))
        labels.append(label)
        paths.append(path.relative_to(root).as_posix())
    return torch.stack(images), torch.tensor(labels), paths


def save_shard(cache: Path, split: str, shard_id: int, latents, labels, paths):
    rel = f"{split}/shard-{shard_id:05d}.pt"
    (cache / split).mkdir(parents=True, exist_ok=True)
    torch.save({"latents": torch.cat(latents), "labels": torch.cat(labels), "source_paths": paths}, cache / rel)
    return {"file": rel, "count": len(paths)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image-root", default="/tmp/data/imagenet-256/ImageNet")
    p.add_argument("--output-root", required=True)
    p.add_argument("--vae", default="stabilityai/sd-vae-ft-ema")
    p.add_argument("--device", default="cpu")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--shard-size", type=int, default=8192)
    p.add_argument("--max-samples", type=int); p.add_argument("--compile-vae", action="store_true")
    args = p.parse_args()
    root, cache = Path(args.image_root), Path(args.output_root)
    transform = build_cache_transform(image_size=256, crop="resize")
    vae = VAEWrapper(model_id=args.vae, latent_shape=[4, 32, 32], device=args.device)
    if args.compile_vae:
        vae.module = torch.compile(vae.module)
    meta = {"cache_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "source_root": os.path.realpath(root), "vae": args.vae, "compiled_vae": args.compile_vae, "transform": {"image_size": 256, "crop": "resize"}, "latent": {"shape": [4, 32, 32], "dtype": "float16"}, "splits": {}}
    for split in args.splits:
        samples = find_samples(root, split, args.max_samples)
        shards, zs, ys, paths, shard_id = [], [], [], [], 0
        for i in tqdm(range(0, len(samples), args.batch_size), desc=split):
            image, label, source_path = load_batch(root, samples[i : i + args.batch_size], transform)
            z = vae.encode(image).detach().cpu().half()
            zs.append(z); ys.append(label); paths += source_path
            if len(paths) >= args.shard_size:
                shards.append(save_shard(cache, split, shard_id, zs, ys, paths))
                zs, ys, paths, shard_id = [], [], [], shard_id + 1
        if paths:
            shards.append(save_shard(cache, split, shard_id, zs, ys, paths))
        meta["splits"][split] = {"count": len(samples), "source_manifest_hash": manifest_hash(samples, root), "shards": shards}
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
