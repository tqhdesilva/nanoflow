#!/usr/bin/env python
"""Benchmark cached ImageNet latent dataloader throughput."""
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
from datasets import ImageNetLatentMMapDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cache-root",
        default=os.environ.get(
            "DATASET_CACHE_ROOT",
            "/tmp/data/imagenet-256-latent-cache/sd-vae-ft-ema-mmap",
        ),
    )
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--warmup-batches", type=int, default=5)
    p.add_argument("--batches", type=int, default=100)
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--transfer-device", default="none", help="none, cpu, cuda, cuda:0, mps")
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def sync_if_needed(device: torch.device) -> None:
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
    train = args.split == "train"
    ds = ImageNetLatentMMapDataset(cache_root=args.cache_root, train=train)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": args.shuffle if train else False,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers if args.num_workers > 0 else False,
        "drop_last": args.drop_last if train else False,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(ds, **loader_kwargs)

    transfer_device = None
    if args.transfer_device != "none":
        transfer_device = torch.device(args.transfer_device)

    total_batches = args.warmup_batches + args.batches
    measured_times: list[float] = []
    measured_samples = 0
    start = None

    it = iter(loader)
    for i in range(total_batches):
        batch_start = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        if transfer_device is not None:
            batch = tuple(
                x.to(transfer_device, non_blocking=True) if isinstance(x, torch.Tensor) else x
                for x in batch
            )
            sync_if_needed(transfer_device)
        batch_time = time.perf_counter() - batch_start
        if i >= args.warmup_batches:
            if start is None:
                start = time.perf_counter() - batch_time
            measured_times.append(batch_time)
            measured_samples += int(batch[0].shape[0])

    elapsed = sum(measured_times)
    result = {
        "cache_root": os.path.realpath(args.cache_root),
        "split": args.split,
        "dataset_samples": len(ds),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": loader_kwargs["persistent_workers"],
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "transfer_device": args.transfer_device,
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
