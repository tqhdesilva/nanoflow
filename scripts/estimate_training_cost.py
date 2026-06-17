#!/usr/bin/env python3
"""Estimate full-run cost from NanoFlow training throughput profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_IMAGENET_TRAIN_SAMPLES = 1_281_167


def load_profile_samples_per_sec(path: Path, *, skip_first: int = 0) -> float:
    """Read a training_profile.jsonl file and return measured samples/sec."""
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    records = records[skip_first:]
    if not records:
        raise ValueError(f"No profile records found in {path}")
    if all("sample_count" in row and "epoch_sec" in row for row in records):
        total_samples = sum(float(row["sample_count"]) for row in records)
        total_seconds = sum(float(row["epoch_sec"]) for row in records)
        if total_samples <= 0 or total_seconds <= 0:
            raise ValueError(f"Profile records in {path} must have positive sample_count and epoch_sec")
        return total_samples / total_seconds
    values = [float(row["samples_per_sec"]) for row in records]
    if not values:
        raise ValueError(f"No samples_per_sec values found in {path}")
    return sum(values) / len(values)


def estimate_cost(
    *,
    samples_per_sec: float,
    epochs: int,
    train_samples: int = DEFAULT_IMAGENET_TRAIN_SAMPLES,
    gpu_hour_price: float,
    num_gpus: int = 1,
) -> dict[str, float | int]:
    """Estimate training wall time and cost from measured throughput."""
    if samples_per_sec <= 0:
        raise ValueError("samples_per_sec must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if train_samples <= 0:
        raise ValueError("train_samples must be positive")
    if gpu_hour_price < 0:
        raise ValueError("gpu_hour_price must be nonnegative")
    if num_gpus <= 0:
        raise ValueError("num_gpus must be positive")
    total_samples = epochs * train_samples
    wall_hours = total_samples / samples_per_sec / 3600.0
    gpu_hours = wall_hours * num_gpus
    cost_usd = gpu_hours * gpu_hour_price
    return {
        "samples_per_sec": samples_per_sec,
        "epochs": epochs,
        "train_samples": train_samples,
        "total_samples": total_samples,
        "wall_hours": wall_hours,
        "num_gpus": num_gpus,
        "gpu_hours": gpu_hours,
        "gpu_hour_price": gpu_hour_price,
        "cost_usd": cost_usd,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--profile", type=Path, help="Path to training_profile.jsonl")
    source.add_argument("--samples-per-sec", type=float, help="Measured samples/sec")
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--train-samples", type=int, default=DEFAULT_IMAGENET_TRAIN_SAMPLES)
    parser.add_argument("--gpu-hour-price", type=float, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    samples_per_sec = args.samples_per_sec
    if args.profile is not None:
        samples_per_sec = load_profile_samples_per_sec(
            args.profile,
            skip_first=args.skip_first,
        )
    result = estimate_cost(
        samples_per_sec=float(samples_per_sec),
        epochs=args.epochs,
        train_samples=args.train_samples,
        gpu_hour_price=args.gpu_hour_price,
        num_gpus=args.num_gpus,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
