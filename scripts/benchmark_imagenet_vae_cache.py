#!/usr/bin/env python
"""Benchmark ImageNet 256 VAE latent cache creation without writing shards."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT))
from scripts.build_imagenet_latent_cache import Images, collate, find_samples
from image_transforms import build_cache_transform
from vae import VAEWrapper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-root", default="/tmp/data/imagenet-256/ImageNet")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--vae", default="stabilityai/sd-vae-ft-ema")
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch-dtype", default="float32")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--max-samples", type=int, default=4096)
    p.add_argument("--warmup-batches", type=int, default=3)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--compile-vae", action="store_true")
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * pct))))
    return values[idx]


def main() -> None:
    args = parse_args()
    root = Path(args.image_root)
    samples = find_samples(root, args.split, args.max_samples)
    transform = build_cache_transform(image_size=256, crop="resize")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers if args.num_workers > 0 else False,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(Images(root, samples, transform), **loader_kwargs)

    device = torch.device(args.device)
    vae = VAEWrapper(
        model_id=args.vae,
        latent_shape=[4, 32, 32],
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    if args.compile_vae:
        vae.module = torch.compile(vae.module)

    measured_times: list[float] = []
    measured_samples = 0
    total_batches = args.warmup_batches + args.batches

    with torch.no_grad():
        it = iter(loader)
        for i in range(total_batches):
            batch_start = time.perf_counter()
            try:
                images, _, _ = next(it)
            except StopIteration:
                break
            z = vae.encode(images)
            sync_device(device)
            if i >= args.warmup_batches:
                measured_times.append(time.perf_counter() - batch_start)
                measured_samples += int(z.shape[0])

    elapsed = sum(measured_times)
    result = {
        "image_root": os.path.realpath(args.image_root),
        "split": args.split,
        "source_samples": len(samples),
        "vae": args.vae,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "compiled_vae": args.compile_vae,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": loader_kwargs["persistent_workers"],
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "measured_batches": len(measured_times),
        "measured_samples": measured_samples,
        "elapsed_sec": elapsed,
        "samples_per_sec": measured_samples / max(elapsed, 1e-9),
        "batches_per_sec": len(measured_times) / max(elapsed, 1e-9),
        "batch_sec_mean": statistics.mean(measured_times) if measured_times else 0.0,
        "batch_sec_p50": percentile(measured_times, 0.50),
        "batch_sec_p95": percentile(measured_times, 0.95),
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
