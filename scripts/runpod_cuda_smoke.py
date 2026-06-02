#!/usr/bin/env python3
"""RunPod CUDA and NanoFlow dependency smoke test."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, TensorDataset

REQUIRED_MODULES = [
    "cleanfid",
    "diffusers",
    "hydra",
    "matplotlib",
    "numpy",
    "omegaconf",
    "PIL",
    "sklearn",
    "tensorboard",
    "torch",
    "torchvision",
    "tqdm",
]


def import_required_modules() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in REQUIRED_MODULES:
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def run_trainer_smoke(device: torch.device) -> dict[str, float | int | str]:
    from config import TrainingConfig
    from flow import CondOT
    from models import MLP
    from train import Trainer

    data = TensorDataset(torch.randn(4, 2))
    loader = DataLoader(data, batch_size=2, shuffle=False)
    training = TrainingConfig(
        epochs=1,
        batch_size=2,
        max_steps=1,
        eval_every=0,
        checkpoint_every=0,
        precision=None,
        ema_decay=0,
    )
    trainer = Trainer(
        model=MLP(hidden_dim=8, num_layers=1, time_dim=8),
        flow=CondOT(),
        training=training,
        device=device,
    )
    trainer.fit(loader, loader, callbacks=[])
    return {
        "device": str(device),
        "steps": trainer.step,
        "last_train_loss": float(trainer.last_train_loss),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow the smoke to run on CPU when CUDA is unavailable.",
    )
    args = parser.parse_args()

    versions = import_required_modules()
    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.allow_cpu:
        print(json.dumps({"cuda_available": False, "versions": versions}, indent=2))
        print("CUDA is not available", file=sys.stderr)
        return 1

    device = torch.device("cuda" if cuda_available else "cpu")
    result: dict[str, object] = {
        "versions": versions,
        "torch_cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
    }

    if cuda_available:
        x = torch.ones(4, device=device)
        y = x * 2
        torch.cuda.synchronize()
        result["cuda_tensor_sum"] = float(y.sum().item())
        result["gpu_name"] = torch.cuda.get_device_name(0)

    result["trainer_smoke"] = run_trainer_smoke(device)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("RunPod CUDA smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
