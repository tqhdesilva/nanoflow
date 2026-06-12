"""Batched ImageNet-256 latent sample generation and FID evaluation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from PIL import Image

import config as _config  # noqa: F401, registers structured config schema


@dataclass(frozen=True)
class CheckpointInfo:
    """Checkpoint details recorded in eval metadata."""

    path: str
    weights: str
    epoch: Optional[int]
    sha256: Optional[str] = None


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for resumable ImageNet latent sample generation.

    Args:
        output_dir: Directory that receives PNGs and sidecar files.
        checkpoint: Checkpoint path used for the model weights.
        num_samples: Number of images to generate.
        batch_size: Max number of missing samples generated per model batch.
        num_steps: Number of endpoint-excluded Euler steps.
        guidance_scale: Classifier-free guidance scale.
        latent_shape: Latent noise shape, without the batch dimension.
        seed: Base seed. Sample index `i` uses seed `seed + i`.
        num_classes: Number of class labels to cycle over.
        image_size: Expected decoded square image size.
        weights: `auto`, `ema`, or `raw` checkpoint weights.
        resume: Whether to skip already written valid PNGs.
        clean_output_dir: Whether to delete the output directory before generation.
    """

    output_dir: Path
    checkpoint: Optional[str]
    num_samples: int = 10000
    batch_size: int = 16
    num_steps: int = 200
    guidance_scale: float = 2.0
    latent_shape: tuple[int, ...] = (4, 32, 32)
    seed: int = 0
    num_classes: int = 1000
    image_size: int = 256
    weights: str = "auto"
    resume: bool = True
    clean_output_dir: bool = False


def build_uniform_labels(num_samples: int, num_classes: int = 1000) -> torch.Tensor:
    """Return deterministic class labels that are balanced when divisible.

    Args:
        num_samples: Number of labels to produce.
        num_classes: Number of classes in the label space.

    Returns:
        Long tensor where label `i` is `i % num_classes`.
    """
    _require_positive_int("num_samples", num_samples)
    _require_positive_int("num_classes", num_classes)
    return torch.arange(num_samples, dtype=torch.long) % int(num_classes)


def endpoint_excluded_euler_grid(
    num_steps: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return Euler times `i / num_steps` for `i = 0..num_steps - 1`.

    Args:
        num_steps: Number of integration steps.
        device: Device for the returned tensor.
        dtype: Floating dtype for the returned tensor.

    Returns:
        A one-dimensional tensor that never includes `t = 1`.
    """
    _require_positive_int("num_steps", num_steps)
    steps = torch.arange(num_steps, device=device, dtype=dtype)
    return steps / float(num_steps)


def load_checkpoint_weights(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    weights: str = "auto",
) -> CheckpointInfo:
    """Load raw or EMA weights into a model.

    Args:
        model: Model to receive weights.
        checkpoint_path: Path to a NanoFlow checkpoint.
        weights: `auto`, `ema`, or `raw`. `auto` prefers EMA when present.

    Returns:
        Information about the loaded weights.
    """
    path = Path(checkpoint_path)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if weights not in {"auto", "ema", "raw"}:
        raise ValueError("weights must be one of: auto, ema, raw")
    if weights == "ema":
        state = ckpt.get("ema_state")
        if state is None:
            raise ValueError(f"Checkpoint has no EMA weights: {path}")
        loaded = "ema"
    elif weights == "raw":
        state = ckpt["model_state"]
        loaded = "raw"
    else:
        state = ckpt.get("ema_state") or ckpt["model_state"]
        loaded = "ema" if ckpt.get("ema_state") else "raw"
    model.load_state_dict(state)
    progress = ckpt.get("train_progress", {})
    epoch = progress.get("num_epochs_completed")
    return CheckpointInfo(
        path=str(path),
        weights=loaded,
        epoch=epoch,
        sha256=sha256_file(path),
    )


@torch.no_grad()
def generate_latents(
    model: torch.nn.Module,
    indices: Sequence[int],
    labels: torch.Tensor,
    *,
    latent_shape: Sequence[int],
    num_steps: int,
    guidance_scale: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a latent batch using endpoint-excluded Euler integration.

    Args:
        model: Class-conditioned velocity model.
        indices: Global sample indices for deterministic per-sample noise.
        labels: Labels for the requested indices.
        latent_shape: Latent tensor shape without batch dimension.
        num_steps: Number of Euler steps.
        guidance_scale: Classifier-free guidance scale.
        seed: Base random seed.
        device: Device used for model evaluation.

    Returns:
        Generated latent tensor with shape `[len(indices), *latent_shape]`.
    """
    if len(indices) == 0:
        raise ValueError("indices must not be empty")
    _require_positive_int("num_steps", num_steps)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    model.eval()
    labels = labels.to(device=device, dtype=torch.long)
    xt = _noise_for_indices(indices, latent_shape, seed).to(device=device)
    dt = 1.0 / float(num_steps)
    for t_val in endpoint_excluded_euler_grid(
        num_steps,
        device=device,
        dtype=xt.dtype,
    ):
        t_batch = t_val.expand(xt.size(0))
        vt = _guided_velocity(model, xt, t_batch, labels, guidance_scale)
        xt = xt + vt * dt
    return xt


def generate_imagenet_samples(
    model: torch.nn.Module,
    vae: Any,
    cfg: GenerationConfig,
    *,
    checkpoint_info: Optional[CheckpointInfo] = None,
    model_config: Optional[Mapping[str, Any]] = None,
    vae_config: Optional[Mapping[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Generate ImageNet eval PNGs and sidecar metadata.

    Args:
        model: Flow model. It is put into eval mode before sampling.
        vae: VAE wrapper with a `decode(latents)` method.
        cfg: Generation settings.
        checkpoint_info: Loaded checkpoint details for metadata.
        model_config: Resolved model config for config-mix protection.
        vae_config: Resolved VAE config for config-mix protection.
        device: Sampling device. Defaults to the first model parameter device.

    Returns:
        Final metadata dictionary written to `metadata.yaml`.
    """
    _validate_generation_config(cfg)
    device = device or _module_device(model)
    output_dir = cfg.output_dir
    critical = _generation_critical_config(
        cfg,
        checkpoint_info=checkpoint_info,
        model_config=model_config,
        vae_config=vae_config,
        vae=vae,
    )
    config_hash = _stable_hash(critical)
    _prepare_output_dir(output_dir, config_hash, cfg.clean_output_dir)

    labels = build_uniform_labels(cfg.num_samples, cfg.num_classes)
    labels_path = _write_labels(output_dir, labels)
    labels_hash = sha256_file(labels_path)
    metadata = _build_generation_metadata(
        cfg,
        config_hash=config_hash,
        critical_config=critical,
        checkpoint_info=checkpoint_info,
        labels_hash=labels_hash,
        vae=vae,
    )
    _write_yaml(output_dir / "metadata.yaml", metadata)

    expected_names = {f"{idx:06d}.png" for idx in range(cfg.num_samples)}
    extra_pngs = [p for p in output_dir.glob("*.png") if p.name not in expected_names]
    if extra_pngs:
        names = ", ".join(sorted(p.name for p in extra_pngs[:5]))
        raise ValueError(
            "Output directory contains PNG files outside the expected sample set: "
            f"{names}. Use clean_output_dir=true or choose a new output_dir."
        )

    for start in range(0, cfg.num_samples, cfg.batch_size):
        batch_indices = list(range(start, min(start + cfg.batch_size, cfg.num_samples)))
        missing = [
            idx
            for idx in batch_indices
            if not (
                cfg.resume
                and is_valid_png(_sample_path(output_dir, idx), cfg.image_size)
            )
        ]
        if not missing:
            continue
        batch_labels = labels[missing]
        latents = generate_latents(
            model,
            missing,
            batch_labels,
            latent_shape=cfg.latent_shape,
            num_steps=cfg.num_steps,
            guidance_scale=cfg.guidance_scale,
            seed=cfg.seed,
            device=device,
        )
        images = vae.decode(latents).detach().cpu()
        _save_image_batch(
            images, [_sample_path(output_dir, idx) for idx in missing], cfg
        )

    valid_count = sum(
        is_valid_png(_sample_path(output_dir, idx), cfg.image_size)
        for idx in range(cfg.num_samples)
    )
    metadata["complete"] = valid_count == cfg.num_samples
    metadata["num_valid_pngs"] = int(valid_count)
    metadata["completed_at"] = _utc_now()
    _write_yaml(output_dir / "metadata.yaml", metadata)
    if valid_count != cfg.num_samples:
        raise RuntimeError(
            f"Expected {cfg.num_samples} valid PNGs, found {valid_count} in {output_dir}"
        )
    return metadata


def is_valid_png(path: str | Path, image_size: int = 256) -> bool:
    """Return true if `path` is a valid RGB PNG of the expected size."""
    path = Path(path)
    if not path.exists() or path.suffix.lower() != ".png":
        return False
    try:
        with Image.open(path) as img:
            img.load()
            return img.mode == "RGB" and img.size == (image_size, image_size)
    except OSError:
        return False


def make_custom_fid_stats(
    image_dir: str | Path,
    *,
    custom_stats_name: str = "nanoflow_imagenet256_val_real_tf_legacy",
    mode: str = "legacy_tensorflow",
    model_name: str = "inception_v3",
    num_workers: int = 8,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    force: bool = False,
    metadata_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Create or reuse Clean-FID custom stats for a folder of real images.

    Args:
        image_dir: Folder containing real ImageNet RGB images.
        custom_stats_name: Clean-FID custom stats name.
        mode: Clean-FID feature mode.
        model_name: Feature model name.
        num_workers: Number of image loader workers.
        batch_size: Feature extraction batch size.
        device: Torch device for feature extraction.
        force: Rebuild stats even if a stats file already exists.
        metadata_path: Optional YAML sidecar path for stats metadata.

    Returns:
        Stats metadata including path, hash, source, and library version.
    """
    from cleanfid import fid

    if image_dir is None:
        raise ValueError("image_dir is required to build Clean-FID stats")
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Real image directory not found: {image_dir}")
    stats_path = cleanfid_stats_path(custom_stats_name, mode, model_name=model_name)
    reused = stats_path.exists() and not force
    if force:
        stats_path.unlink(missing_ok=True)
        cleanfid_stats_path(
            custom_stats_name,
            mode,
            model_name=model_name,
            metric="KID",
        ).unlink(missing_ok=True)
    if not reused:
        fid.make_custom_stats(
            custom_stats_name,
            str(image_dir),
            mode=mode,
            model_name=model_name,
            num_workers=num_workers,
            batch_size=batch_size,
            device=torch.device(device),
        )
    if not stats_path.exists():
        raise RuntimeError(f"Clean-FID did not write stats file: {stats_path}")
    metadata = {
        "kind": "nanoflow_imagenet256_fid_stats",
        "custom_stats_name": custom_stats_name,
        "real_image_source": str(image_dir),
        "mode": mode,
        "model_name": model_name,
        "feature_extractor": _feature_extractor_name(model_name),
        "stats_path": str(stats_path),
        "stats_sha256": sha256_file(stats_path),
        "cleanfid_version": _package_version("clean-fid"),
        "reused": bool(reused),
        "created_at": _utc_now(),
    }
    if metadata_path is not None:
        _write_yaml(Path(metadata_path), metadata)
    return metadata


def compute_imagenet_fid(
    sample_dir: str | Path,
    *,
    custom_stats_name: Optional[str] = "nanoflow_imagenet256_val_real_tf_legacy",
    dataset_name: Optional[str] = None,
    dataset_res: int | str = 256,
    dataset_split: str = "val",
    mode: str = "legacy_tensorflow",
    model_name: str = "inception_v3",
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    num_workers: int = 8,
    expected_num_samples: Optional[int] = None,
    output_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Compute Clean-FID for generated ImageNet PNGs.

    Args:
        sample_dir: Directory with generated PNG files.
        custom_stats_name: Custom Clean-FID stats name. When set, Clean-FID uses
            `split=custom` and `res=na`.
        dataset_name: Built-in Clean-FID dataset name when custom stats are not used.
        dataset_res: Built-in dataset resolution.
        dataset_split: Built-in dataset split.
        mode: Clean-FID feature mode.
        model_name: Clean-FID feature model name.
        device: Torch device for feature extraction.
        batch_size: Feature extraction batch size.
        num_workers: Number of image loader workers.
        expected_num_samples: Optional exact PNG count to require before scoring.
        output_path: Optional YAML path for the metric result.

    Returns:
        Metric result dictionary.
    """
    from cleanfid import fid

    sample_dir = Path(sample_dir)
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    n_samples = _count_pngs(sample_dir)
    if n_samples == 0:
        raise ValueError(f"Sample directory has no PNG files: {sample_dir}")
    if expected_num_samples is not None and n_samples != int(expected_num_samples):
        raise ValueError(
            f"Expected {expected_num_samples} PNG samples, found {n_samples} in {sample_dir}"
        )

    if custom_stats_name:
        ref_name = custom_stats_name
        ref_res: int | str = "na"
        ref_split = "custom"
        stats_path = cleanfid_stats_path(custom_stats_name, mode, model_name=model_name)
        stats_hash = sha256_file(stats_path) if stats_path.exists() else None
    else:
        if dataset_name is None:
            raise ValueError("dataset_name is required when custom_stats_name is null")
        ref_name = dataset_name
        ref_res = dataset_res
        ref_split = dataset_split
        stats_path = None
        stats_hash = None

    score = fid.compute_fid(
        str(sample_dir),
        dataset_name=ref_name,
        dataset_res=ref_res,
        dataset_split=ref_split,
        mode=mode,
        model_name=model_name,
        device=torch.device(device),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    result = {
        "name": "FID",
        "score": float(score),
        "sample_dir": str(sample_dir),
        "num_samples": int(n_samples),
        "expected_num_samples": (
            None if expected_num_samples is None else int(expected_num_samples)
        ),
        "mode": mode,
        "model_name": model_name,
        "feature_extractor": _feature_extractor_name(model_name),
        "dataset_name": ref_name,
        "dataset_res": ref_res,
        "dataset_split": ref_split,
        "custom_stats_name": custom_stats_name,
        "stats_path": str(stats_path) if stats_path is not None else None,
        "stats_sha256": stats_hash,
        "cleanfid_version": _package_version("clean-fid"),
    }
    if output_path is not None:
        _write_yaml(Path(output_path), result)
    return result


def cleanfid_stats_path(
    name: str,
    mode: str,
    *,
    model_name: str = "inception_v3",
    metric: str = "FID",
) -> Path:
    """Return the local Clean-FID stats path for a custom stats name."""
    import cleanfid

    stats_folder = Path(cleanfid.__file__).resolve().parent / "stats"
    model_modifier = "" if model_name == "inception_v3" else f"_{model_name}"
    kid_suffix = "_kid" if metric == "KID" else ""
    filename = f"{name}_{mode}{model_modifier}_custom_na{kid_suffix}.npz".lower()
    return stats_folder / filename


def sha256_file(path: str | Path) -> str:
    """Return a hex SHA256 digest for a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guided_velocity(
    model: torch.nn.Module,
    xt: torch.Tensor,
    t_batch: torch.Tensor,
    labels: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Return one classifier-free-guided velocity batch."""
    if float(guidance_scale) == 1.0:
        return model(xt, t_batch, labels)
    if not hasattr(model, "null_token"):
        raise ValueError("CFG sampling requires model.null_token")
    null_labels = torch.full_like(labels, int(getattr(model, "null_token")))
    v_cond = model(xt, t_batch, labels)
    v_uncond = model(xt, t_batch, null_labels)
    return v_uncond + float(guidance_scale) * (v_cond - v_uncond)


def _noise_for_indices(
    indices: Sequence[int],
    latent_shape: Sequence[int],
    seed: int,
) -> torch.Tensor:
    """Return deterministic CPU noise using `seed + sample_index`."""
    latents = []
    for idx in indices:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + int(idx))
        latents.append(torch.randn(*tuple(latent_shape), generator=gen))
    return torch.stack(latents, dim=0)


def _save_image_batch(
    images: torch.Tensor,
    paths: Sequence[Path],
    cfg: GenerationConfig,
) -> None:
    if images.dim() != 4 or images.size(1) != 3:
        raise ValueError(
            f"Expected decoded images [N, 3, H, W], got {tuple(images.shape)}"
        )
    if images.shape[-2:] != (cfg.image_size, cfg.image_size):
        raise ValueError(
            f"Expected decoded image size {cfg.image_size}, got {tuple(images.shape[-2:])}"
        )
    if images.size(0) != len(paths):
        raise ValueError("Number of images does not match number of output paths")
    for image, path in zip(images, paths):
        pil_image = _tensor_to_rgb_image(image)
        _save_png_atomic(pil_image, path)


def _tensor_to_rgb_image(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(-1, 1)
    image = (image * 0.5 + 0.5) * 255.0
    array = image.round().to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(np.asarray(array))


def _save_png_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    image.save(tmp_path, format="PNG")
    tmp_path.replace(path)


def _sample_path(output_dir: Path, idx: int) -> Path:
    return output_dir / f"{idx:06d}.png"


def _write_labels(output_dir: Path, labels: torch.Tensor) -> Path:
    records = [
        {"index": int(idx), "filename": f"{idx:06d}.png", "label": int(label)}
        for idx, label in enumerate(labels.tolist())
    ]
    payload = {
        "num_samples": int(labels.numel()),
        "label_schedule": "cycle_mod_num_classes",
        "records": records,
    }
    path = output_dir / "labels.json"
    _write_json(path, payload)
    return path


def _build_generation_metadata(
    cfg: GenerationConfig,
    *,
    config_hash: str,
    critical_config: Mapping[str, Any],
    checkpoint_info: Optional[CheckpointInfo],
    labels_hash: str,
    vae: Any,
) -> dict[str, Any]:
    return {
        "kind": "nanoflow_imagenet256_eval_samples",
        "created_at": _utc_now(),
        "complete": False,
        "num_valid_pngs": 0,
        "config_hash": config_hash,
        "config": dict(critical_config),
        "checkpoint": asdict(checkpoint_info) if checkpoint_info is not None else None,
        "git": {"commit": _git_commit()},
        "sampler": {
            "type": "ode_euler",
            "time_grid": "endpoint_excluded",
            "num_steps": int(cfg.num_steps),
            "dt": 1.0 / float(cfg.num_steps),
        },
        "cfg": {"guidance_scale": float(cfg.guidance_scale)},
        "seed": int(cfg.seed),
        "noise_schedule": "per_index_seed",
        "labels": {
            "file": "labels.json",
            "sha256": labels_hash,
            "num_classes": int(cfg.num_classes),
            "schedule": "cycle_mod_num_classes",
        },
        "vae": {
            "id": getattr(vae, "model_id", type(vae).__name__),
            "image_size": int(cfg.image_size),
        },
        "image_format": "png_uint8_rgb",
    }


def _generation_critical_config(
    cfg: GenerationConfig,
    *,
    checkpoint_info: Optional[CheckpointInfo],
    model_config: Optional[Mapping[str, Any]],
    vae_config: Optional[Mapping[str, Any]],
    vae: Any,
) -> dict[str, Any]:
    return {
        "checkpoint": _checkpoint_identity(checkpoint_info, cfg.checkpoint),
        "checkpoint_sha256": (
            checkpoint_info.sha256 if checkpoint_info is not None else None
        ),
        "checkpoint_epoch": (
            checkpoint_info.epoch if checkpoint_info is not None else None
        ),
        "weights": (
            checkpoint_info.weights if checkpoint_info is not None else cfg.weights
        ),
        "num_samples": int(cfg.num_samples),
        "num_steps": int(cfg.num_steps),
        "guidance_scale": float(cfg.guidance_scale),
        "latent_shape": list(cfg.latent_shape),
        "seed": int(cfg.seed),
        "num_classes": int(cfg.num_classes),
        "image_size": int(cfg.image_size),
        "model_config": _jsonable(model_config),
        "vae_config": _jsonable(vae_config),
        "vae_id": getattr(vae, "model_id", type(vae).__name__),
        "sampler": "endpoint_excluded_euler",
        "image_format": "png_uint8_rgb",
    }


def _checkpoint_identity(
    checkpoint_info: Optional[CheckpointInfo],
    checkpoint: Optional[str],
) -> Optional[str]:
    path = checkpoint_info.path if checkpoint_info is not None else checkpoint
    if path is None:
        return None
    path_obj = Path(path)
    return str(path_obj.resolve()) if path_obj.exists() else str(path_obj)


def _prepare_output_dir(
    output_dir: Path, config_hash: str, clean_output_dir: bool
) -> None:
    """Create an output directory and reject unsafe config mixing."""
    if clean_output_dir and output_dir.exists():
        _assert_safe_clean_output_dir(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path) as handle:
            existing = yaml.safe_load(handle) or {}
        if existing.get("config_hash") != config_hash:
            raise ValueError(
                "Output directory already has metadata for a different eval config. "
                "Use clean_output_dir=true or choose a new output_dir."
            )
    elif any(output_dir.glob("*.png")):
        raise ValueError(
            "Output directory contains PNG files but no metadata.yaml. "
            "Use clean_output_dir=true or choose a new output_dir."
        )


def _assert_safe_clean_output_dir(output_dir: Path) -> None:
    """Reject broad or non-eval paths before recursive deletion."""
    resolved = output_dir.expanduser().resolve()
    forbidden = {Path("/").resolve(), Path.cwd().resolve(), Path.home().resolve()}
    if resolved in forbidden or len(resolved.parts) <= 2:
        raise ValueError(f"Refusing to clean unsafe output_dir: {output_dir}")
    metadata_path = output_dir / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path) as handle:
            metadata = yaml.safe_load(handle) or {}
        if metadata.get("kind") != "nanoflow_imagenet256_eval_samples":
            raise ValueError(
                "Refusing to clean output_dir with non-eval metadata.yaml: "
                f"{output_dir}"
            )
        return
    if any(output_dir.iterdir()):
        raise ValueError(
            "Refusing to clean a non-empty output_dir without metadata.yaml: "
            f"{output_dir}"
        )


def _validate_generation_config(cfg: GenerationConfig) -> None:
    _require_positive_int("num_samples", cfg.num_samples)
    _require_positive_int("batch_size", cfg.batch_size)
    _require_positive_int("num_steps", cfg.num_steps)
    _require_positive_int("num_classes", cfg.num_classes)
    _require_positive_int("image_size", cfg.image_size)
    if cfg.seed < 0:
        raise ValueError("seed must be nonnegative")
    if cfg.weights not in {"auto", "ema", "raw"}:
        raise ValueError("weights must be one of: auto, ema, raw")
    if not cfg.latent_shape:
        raise ValueError("latent_shape must not be empty")
    for dim in cfg.latent_shape:
        _require_positive_int("latent_shape dim", dim)


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _feature_extractor_name(model_name: str) -> str:
    if model_name == "inception_v3":
        return "inception_v3_pool3"
    return model_name


def _count_pngs(path: Path) -> int:
    return sum(1 for _ in path.glob("*.png"))


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w") as handle:
        yaml.dump(_jsonable(payload), handle, default_flow_style=False, sort_keys=False)
    tmp_path.replace(path)


def _git_commit() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _package_version(package: str) -> Optional[str]:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    return OmegaConf.select(cfg, key, default=default)


def _generation_config_from_hydra(eval_cfg: Any) -> GenerationConfig:
    """Convert the structured Hydra eval node to `GenerationConfig`."""
    output_dir = _cfg_get(eval_cfg, "output_dir")
    if output_dir is None:
        raise ValueError("eval.output_dir is required for generation")
    return GenerationConfig(
        output_dir=Path(output_dir),
        checkpoint=_cfg_get(eval_cfg, "checkpoint"),
        num_samples=int(_cfg_get(eval_cfg, "generation.num_samples", 10000)),
        batch_size=int(_cfg_get(eval_cfg, "generation.batch_size", 16)),
        num_steps=int(_cfg_get(eval_cfg, "generation.num_steps", 200)),
        guidance_scale=float(_cfg_get(eval_cfg, "generation.guidance_scale", 2.0)),
        latent_shape=tuple(_cfg_get(eval_cfg, "generation.latent_shape", [4, 32, 32])),
        seed=int(_cfg_get(eval_cfg, "seed", 0)),
        num_classes=int(_cfg_get(eval_cfg, "generation.num_classes", 1000)),
        image_size=int(_cfg_get(eval_cfg, "generation.image_size", 256)),
        weights=str(_cfg_get(eval_cfg, "weights", "auto")),
        resume=bool(_cfg_get(eval_cfg, "generation.resume", True)),
        clean_output_dir=bool(_cfg_get(eval_cfg, "generation.clean_output_dir", False)),
    )


def _output_dir_or_none(eval_cfg: Any) -> Optional[Path]:
    output_dir = _cfg_get(eval_cfg, "output_dir")
    return None if output_dir is None else Path(output_dir)


@hydra.main(config_path="configs", config_name="eval_imagenet", version_base=None)
def main(cfg) -> None:
    """Hydra entry point for ImageNet sample generation and FID."""
    eval_cfg = cfg.eval
    device = torch.device(_cfg_get(eval_cfg, "device", cfg.device))
    output_dir = _output_dir_or_none(eval_cfg)

    if bool(_cfg_get(eval_cfg, "make_stats", False)):
        stats_metadata_path = _cfg_get(eval_cfg, "stats.metadata_path")
        if stats_metadata_path is None and output_dir is not None:
            stats_metadata_path = output_dir / "stats_metadata.yaml"
        make_custom_fid_stats(
            _cfg_get(eval_cfg, "stats.real_dir"),
            custom_stats_name=_cfg_get(
                eval_cfg,
                "stats.custom_stats_name",
                "nanoflow_imagenet256_val_real_tf_legacy",
            ),
            mode=_cfg_get(eval_cfg, "stats.mode", "legacy_tensorflow"),
            model_name=_cfg_get(eval_cfg, "stats.model_name", "inception_v3"),
            num_workers=int(_cfg_get(eval_cfg, "stats.num_workers", 8)),
            batch_size=int(_cfg_get(eval_cfg, "stats.batch_size", 64)),
            device=_cfg_get(eval_cfg, "stats.device", str(device)),
            force=bool(_cfg_get(eval_cfg, "stats.force", False)),
            metadata_path=stats_metadata_path,
        )

    if bool(_cfg_get(eval_cfg, "generate", True)):
        gen_cfg = _generation_config_from_hydra(eval_cfg)
        if gen_cfg.checkpoint is None:
            raise ValueError("eval.checkpoint is required for generation")
        model = hydra.utils.instantiate(cfg.model)
        checkpoint_info = load_checkpoint_weights(
            model, gen_cfg.checkpoint, gen_cfg.weights
        )
        model.to(device)
        model.eval()
        vae = hydra.utils.instantiate(cfg.vae, device=str(device))
        generate_imagenet_samples(
            model,
            vae,
            gen_cfg,
            checkpoint_info=checkpoint_info,
            model_config=_jsonable(cfg.model),
            vae_config=_jsonable(cfg.vae),
            device=device,
        )

    if bool(_cfg_get(eval_cfg, "compute_fid", False)):
        fid_sample_dir = _cfg_get(eval_cfg, "fid.sample_dir")
        if fid_sample_dir is None:
            if output_dir is None:
                raise ValueError("eval.fid.sample_dir or eval.output_dir is required")
            fid_sample_dir = output_dir
        metrics_path = _cfg_get(eval_cfg, "fid.output_path")
        if metrics_path is None and output_dir is not None:
            metrics_path = output_dir / "metrics.yaml"
        compute_imagenet_fid(
            fid_sample_dir,
            custom_stats_name=_cfg_get(
                eval_cfg,
                "fid.custom_stats_name",
                "nanoflow_imagenet256_val_real_tf_legacy",
            ),
            dataset_name=_cfg_get(eval_cfg, "fid.dataset_name"),
            dataset_res=_cfg_get(eval_cfg, "fid.dataset_res", 256),
            dataset_split=_cfg_get(eval_cfg, "fid.dataset_split", "val"),
            mode=_cfg_get(eval_cfg, "fid.mode", "legacy_tensorflow"),
            model_name=_cfg_get(eval_cfg, "fid.model_name", "inception_v3"),
            device=_cfg_get(eval_cfg, "fid.device", str(device)),
            batch_size=int(_cfg_get(eval_cfg, "fid.batch_size", 64)),
            num_workers=int(_cfg_get(eval_cfg, "fid.num_workers", 8)),
            expected_num_samples=_cfg_get(eval_cfg, "fid.num_samples"),
            output_path=metrics_path,
        )


if __name__ == "__main__":
    main()
