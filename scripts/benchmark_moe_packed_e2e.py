#!/usr/bin/env python3
"""Benchmark full-model training steps with loop vs packed MoE FFNs."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.nn.utils import clip_grad_norm_

import config as _config  # noqa: F401, registers config schemas
from train import setup_device_and_dist

LOOP_TARGET = "models_dit.ExpertChoiceMoEFFN"
PACKED_TARGET = "models_dit.PackedExpertChoiceMoEFFN"


@dataclass(frozen=True)
class VariantResult:
    name: str
    experiment: str
    batch_size: int
    warmup_steps: int
    measured_steps: int
    replaced_moe_targets: int
    step_ms_mean: float
    step_ms_median: float
    fetch_ms_mean: float
    total_ms_mean: float
    samples_per_sec_model: float
    samples_per_sec_with_fetch: float
    peak_memory_gib: float | None

    def row(self) -> dict[str, Any]:
        return self.__dict__.copy()


def compose_cfg(experiment: str, overrides: list[str]) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"), version_base=None
    ):
        return compose(
            config_name="config", overrides=[f"experiment={experiment}", *overrides]
        )


def replace_moe_targets(node: Any, target: str) -> int:
    replacements = 0
    if isinstance(node, ListConfig):
        for item in node:
            replacements += replace_moe_targets(item, target)
        return replacements
    if not isinstance(node, DictConfig):
        return 0
    if node.get("_target_") == LOOP_TARGET:
        node._target_ = target
        replacements += 1
    for value in node.values():
        replacements += replace_moe_targets(value, target)
    return replacements


def normalize_cfg(cfg: DictConfig, args: argparse.Namespace, variant: str) -> int:
    OmegaConf.set_struct(cfg, False)
    cfg.device = args.device
    cfg.distributed = None
    cfg.training.batch_size = args.batch_size
    cfg.training.num_workers = args.num_workers
    cfg.training.resume = None
    cfg.training.epochs = 1
    cfg.training.max_steps = args.warmup_steps + args.steps
    cfg.training.eval_every = 0
    cfg.training.checkpoint_every = 0
    cfg.training.log_every = max(args.steps + args.warmup_steps, 1)
    cfg.training.run_dir = None
    if args.precision is not None:
        cfg.training.precision = args.precision
    if args.disable_ema:
        cfg.training.ema_decay = 0
    cfg.sample_logger = None
    cfg.inference = None
    cfg.train_loader.persistent_workers = args.persistent_workers
    cfg.val_loader.persistent_workers = False
    if args.dataset_cache_root is not None:
        cfg.dataset.cache_root = args.dataset_cache_root
    target = PACKED_TARGET if variant == "packed" else LOOP_TARGET
    return replace_moe_targets(cfg.model, target)


@contextlib.contextmanager
def cuda_peak_memory(device: torch.device) -> Iterator[None]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    yield


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_one_step(trainer, batch) -> None:
    trainer.optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(
        device_type=trainer.device.type,
        dtype=trainer._amp_dtype,
        enabled=trainer._amp,
    ):
        loss, _ = trainer._compute_loss(batch)
    if trainer.scaler is not None:
        trainer.scaler.scale(loss).backward()
        trainer.scaler.unscale_(trainer.optimizer)
        if trainer.training.grad_clip > 0:
            clip_grad_norm_(trainer.module.parameters(), trainer.training.grad_clip)
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
    else:
        loss.backward()
        if trainer.training.grad_clip > 0:
            clip_grad_norm_(trainer.module.parameters(), trainer.training.grad_clip)
        trainer.optimizer.step()
    if trainer.ema_model is not None:
        trainer.ema_model.update_parameters(trainer.raw_module)


def benchmark_variant(
    base_cfg: DictConfig,
    args: argparse.Namespace,
    *,
    variant: str,
) -> VariantResult:
    cfg = copy.deepcopy(base_cfg)
    replaced = normalize_cfg(cfg, args, variant)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = setup_device_and_dist(cfg.device, cfg.distributed)
    trainer = hydra.utils.instantiate(cfg.trainer, device=device)
    train_loader = hydra.utils.instantiate(cfg.train_loader)
    trainer.module.train()

    iterator = iter(train_loader)
    step_times = []
    fetch_times = []
    total_times = []
    total_steps = args.warmup_steps + args.steps
    with cuda_peak_memory(device):
        for step_idx in range(total_steps):
            total_start = time.perf_counter()
            fetch_start = time.perf_counter()
            batch, iterator = next_batch(iterator, train_loader)
            fetch_end = time.perf_counter()
            batch = trainer._to_device(batch)
            step_start = time.perf_counter()
            train_one_step(trainer, batch)
            sync_device(device)
            step_end = time.perf_counter()
            if step_idx >= args.warmup_steps:
                fetch_times.append(fetch_end - fetch_start)
                step_times.append(step_end - step_start)
                total_times.append(step_end - total_start)
    if hasattr(train_loader, "_iterator") and train_loader._iterator is not None:
        train_loader._iterator._shutdown_workers()
    peak_memory_gib = None
    if device.type == "cuda":
        peak_memory_gib = torch.cuda.max_memory_allocated(device) / 2**30
    batch_size = int(cfg.training.batch_size)
    step_mean = statistics.mean(step_times)
    total_mean = statistics.mean(total_times)
    return VariantResult(
        name=variant,
        experiment=args.experiment,
        batch_size=batch_size,
        warmup_steps=args.warmup_steps,
        measured_steps=args.steps,
        replaced_moe_targets=replaced,
        step_ms_mean=step_mean * 1000.0,
        step_ms_median=statistics.median(step_times) * 1000.0,
        fetch_ms_mean=statistics.mean(fetch_times) * 1000.0,
        total_ms_mean=total_mean * 1000.0,
        samples_per_sec_model=batch_size / step_mean,
        samples_per_sec_with_fetch=batch_size / total_mean,
        peak_memory_gib=peak_memory_gib,
    )


def speedup(loop: VariantResult, packed: VariantResult) -> dict[str, float]:
    return {
        "model_step_speedup": loop.step_ms_mean / packed.step_ms_mean,
        "total_step_speedup": loop.total_ms_mean / packed.total_ms_mean,
        "samples_per_sec_model_speedup": packed.samples_per_sec_model
        / loop.samples_per_sec_model,
        "samples_per_sec_with_fetch_speedup": packed.samples_per_sec_with_fetch
        / loop.samples_per_sec_with_fetch,
    }


def print_markdown(results: list[VariantResult], summary: dict[str, float]) -> None:
    print(
        "| Variant | MoE targets | Step ms | Total ms | Model sam/s | Total sam/s | Peak GiB |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        peak = "" if result.peak_memory_gib is None else f"{result.peak_memory_gib:.2f}"
        print(
            "| {name} | {targets} | {step:.2f} | {total:.2f} | {model_sps:.2f} | {total_sps:.2f} | {peak} |".format(
                name=result.name,
                targets=result.replaced_moe_targets,
                step=result.step_ms_mean,
                total=result.total_ms_mean,
                model_sps=result.samples_per_sec_model,
                total_sps=result.samples_per_sec_with_fetch,
                peak=peak,
            )
        )
    print()
    print("| Metric | Speedup |")
    print("| --- | ---: |")
    for key, value in summary.items():
        print(f"| {key} | {value:.3f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", default="imagenet256_latent_dit_b2_moe_layerwise"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dataset-cache-root", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--disable-ema", action="store_true")
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    base_cfg = compose_cfg(args.experiment, args.overrides)
    results = [
        benchmark_variant(base_cfg, args, variant="loop"),
        benchmark_variant(base_cfg, args, variant="packed"),
    ]
    summary = speedup(results[0], results[1])
    payload = {
        "results": [result.row() for result in results],
        "speedup": summary,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        with (
            args.markdown_output.open("w") as handle,
            contextlib.redirect_stdout(handle),
        ):
            print_markdown(results, summary)
    print_markdown(results, summary)


if __name__ == "__main__":
    main()
