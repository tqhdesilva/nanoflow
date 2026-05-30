"""VAE integration for latent ImageNet training.

The wrapper expects RGB tensors in [-1, 1]. Encoded latents include the VAE
scaling factor, so decoding divides by the same factor before calling the VAE.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _dtype_from_name(name: Optional[str]) -> torch.dtype:
    if name in (None, "float32", "fp32"):
        return torch.float32
    if name in ("float16", "fp16"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unknown torch_dtype: {name!r}")


def _import_autoencoder_kl():
    try:
        from diffusers.models import AutoencoderKL
    except ImportError as exc:
        raise ImportError(
            "diffusers is required for VAE integration. Install with `uv add diffusers`."
        ) from exc
    return AutoencoderKL


class VAEWrapper:
    """Thin wrapper around supported ImageNet 256 VAE backends."""

    def __init__(
        self,
        model_id: str = "stabilityai/sd-vae-ft-ema",
        backend: str = "diffusers_autoencoder_kl",
        revision: Optional[str] = None,
        subfolder: Optional[str] = None,
        latent_shape: Optional[list[int]] = None,
        image_size: int = 256,
        scaling_factor: Optional[float] = None,
        torch_dtype: Optional[str] = "float32",
        device: str = "cpu",
        sample_posterior: bool = False,
        local_files_only: bool = False,
        load_model: bool = True,
    ):
        self.model_id = model_id
        self.backend = backend
        self.revision = revision
        self.subfolder = subfolder
        self.latent_shape = tuple(latent_shape) if latent_shape is not None else None
        self.image_size = image_size
        self.scaling_factor = scaling_factor
        self.torch_dtype = _dtype_from_name(torch_dtype)
        self.device = torch.device(device)
        self.sample_posterior = sample_posterior
        self.local_files_only = local_files_only
        self.module = None
        if load_model:
            self.load()

    def load(self):
        if self.module is not None:
            return self
        if self.backend != "diffusers_autoencoder_kl":
            raise NotImplementedError(
                f"VAE backend {self.backend!r} is registered in config but not yet "
                "implemented. Use backend='diffusers_autoencoder_kl' for Stage 2.2."
            )
        AutoencoderKL = _import_autoencoder_kl()
        kwargs = {
            "pretrained_model_name_or_path": self.model_id,
            "torch_dtype": self.torch_dtype,
            "local_files_only": self.local_files_only,
        }
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.subfolder is not None:
            kwargs["subfolder"] = self.subfolder
        self.module = AutoencoderKL.from_pretrained(**kwargs)
        self.module.to(device=self.device, dtype=self.torch_dtype)
        self.module.eval()
        if self.scaling_factor is None:
            self.scaling_factor = float(
                getattr(getattr(self.module, "config", None), "scaling_factor", 1.0)
            )
        return self

    def _assert_loaded(self):
        if self.module is None:
            self.load()

    def _assert_latent_shape(self, z: torch.Tensor) -> None:
        if self.latent_shape is None:
            return
        if tuple(z.shape[1:]) != self.latent_shape:
            raise ValueError(
                f"Encoded latent shape {tuple(z.shape[1:])} does not match "
                f"configured latent_shape {self.latent_shape}"
            )

    @torch.no_grad()
    def encode(self, x: torch.Tensor, sample_posterior: Optional[bool] = None):
        """Encode RGB images in [-1, 1] to scaled latents."""
        self._assert_loaded()
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected image batch [N, 3, H, W], got {tuple(x.shape)}")
        x = x.to(device=self.device, dtype=self.torch_dtype)
        posterior = self.module.encode(x).latent_dist
        sample = self.sample_posterior if sample_posterior is None else sample_posterior
        if sample:
            z = posterior.sample()
        elif hasattr(posterior, "mode"):
            z = posterior.mode()
        else:
            z = posterior.mean
        z = z * float(self.scaling_factor)
        self._assert_latent_shape(z)
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor, clamp: bool = True):
        """Decode scaled latents back to RGB tensors in [-1, 1]."""
        self._assert_loaded()
        if z.dim() != 4:
            raise ValueError(
                f"Expected latent batch [N, C, H, W], got {tuple(z.shape)}"
            )
        self._assert_latent_shape(z)
        z = z.to(device=self.device, dtype=self.torch_dtype)
        y = self.module.decode(z / float(self.scaling_factor)).sample
        if y.shape[-2:] != (self.image_size, self.image_size):
            y = F.interpolate(
                y,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        if clamp:
            y = y.clamp(-1, 1)
        return y
