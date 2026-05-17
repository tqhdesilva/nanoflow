"""Generation metrics, polymorphic and instantiated via Hydra `_target_`.

Each metric is a callable taking a `(N, C, H, W)` tensor in `[-1, 1]` and
returning a result dict that `main.py` aggregates into the run dir.
"""

import tempfile
from pathlib import Path

import torch
from torchvision.utils import save_image

from rl.compression import jpeg_bytes_for_sample, png_bytes_for_sample


def _summary(values: torch.Tensor, prefix: str) -> dict:
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()),
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
    }


class JpegCompressibilityMetric:
    """JPEG compressibility metrics for samples in `[-1, 1]`."""

    def __init__(
        self,
        quality: int = 75,
        optimize: bool = False,
        progressive: bool = False,
        subsampling: int | str | None = None,
        include_png: bool = False,
        include_diagnostics: bool = False,
    ):
        self.quality = quality
        self.optimize = optimize
        self.progressive = progressive
        self.subsampling = subsampling
        self.include_png = include_png
        self.include_diagnostics = include_diagnostics

    def __call__(self, samples: torch.Tensor) -> dict:
        imgs = samples.detach().cpu()
        h, w = int(imgs.shape[-2]), int(imgs.shape[-1])
        jpeg_bytes = torch.tensor(
            [
                jpeg_bytes_for_sample(
                    img,
                    quality=self.quality,
                    optimize=self.optimize,
                    progressive=self.progressive,
                    subsampling=self.subsampling,
                )
                for img in imgs
            ],
            dtype=torch.float32,
        )
        jpeg_bpp = 8.0 * jpeg_bytes / float(h * w)

        result = {
            "name": "JPEGCompressibility",
            "n_samples": int(imgs.shape[0]),
            "quality": int(self.quality),
            "optimize": bool(self.optimize),
            "progressive": bool(self.progressive),
            "subsampling": self.subsampling,
            "jpeg_bytes": [int(x) for x in jpeg_bytes.tolist()],
            **_summary(jpeg_bytes, "jpeg_bytes"),
            **_summary(jpeg_bpp, "jpeg_bpp"),
        }

        if self.include_png:
            png_bytes = torch.tensor(
                [png_bytes_for_sample(img) for img in imgs], dtype=torch.float32
            )
            png_bpp = 8.0 * png_bytes / float(h * w)
            result.update(
                {
                    "png_bytes": [int(x) for x in png_bytes.tolist()],
                    **_summary(png_bytes, "png_bytes"),
                    **_summary(png_bpp, "png_bpp"),
                }
            )

        if self.include_diagnostics:
            clamped = imgs.clamp(-1, 1)
            tv_h = (
                (clamped[..., 1:, :] - clamped[..., :-1, :])
                .abs()
                .mean(dim=(-3, -2, -1))
            )
            tv_w = (
                (clamped[..., :, 1:] - clamped[..., :, :-1])
                .abs()
                .mean(dim=(-3, -2, -1))
            )
            total_variation = tv_h + tv_w
            centered = clamped - clamped.mean(dim=(-2, -1), keepdim=True)
            spectrum = torch.fft.rfft2(centered, norm="ortho").abs().pow(2)
            yy = torch.arange(h, dtype=torch.float32).view(h, 1)
            xx = torch.arange(spectrum.shape[-1], dtype=torch.float32).view(1, -1)
            high_mask = (
                yy / max(h - 1, 1) + xx / max(spectrum.shape[-1] - 1, 1)
            ) > 0.75
            high_freq_energy = spectrum[..., high_mask].mean(dim=(-1,))
            flat = ((clamped + 1.0) * 127.5).round().view(clamped.size(0), -1)
            duplicate_count = int(torch.unique(flat, dim=0).size(0))
            result.update(
                {
                    **_summary(total_variation, "total_variation"),
                    **_summary(high_freq_energy, "high_freq_energy"),
                    "unique_samples": duplicate_count,
                    "duplicate_fraction": float(
                        1.0 - duplicate_count / max(int(clamped.size(0)), 1)
                    ),
                }
            )

        print(
            f"[JPEG] q={self.quality} n={result['n_samples']} "
            f"bpp={result['jpeg_bpp_mean']:.4f}±{result['jpeg_bpp_std']:.4f}",
            flush=True,
        )
        return result


class FIDMetric:
    """Clean-FID against a precomputed reference dataset (Parmar et al., 2022)."""

    def __init__(
        self,
        dataset_name: str = "cifar10",
        dataset_res: int = 32,
        dataset_split: str = "train",
        mode: str = "clean",
        device: str = "cpu",
        num_workers: int = 0,
    ):
        self.dataset_name = dataset_name
        self.dataset_res = dataset_res
        self.dataset_split = dataset_split
        self.mode = mode
        self.device = torch.device(device)
        self.num_workers = num_workers

    def __call__(self, samples: torch.Tensor) -> dict:
        from cleanfid import fid

        imgs = (samples.clamp(-1, 1) * 0.5 + 0.5).cpu()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for i, img in enumerate(imgs):
                save_image(img, tmp_path / f"{i:06d}.png")
            score = fid.compute_fid(
                str(tmp_path),
                dataset_name=self.dataset_name,
                dataset_res=self.dataset_res,
                dataset_split=self.dataset_split,
                mode=self.mode,
                device=self.device,
                num_workers=self.num_workers,
            )
        result = {
            "name": "FID",
            "dataset_name": self.dataset_name,
            "dataset_split": self.dataset_split,
            "mode": self.mode,
            "n_samples": int(imgs.shape[0]),
            "score": float(score),
        }
        print(
            f"[FID] {self.dataset_name}/{self.dataset_split} "
            f"({self.mode}, n={result['n_samples']}): {result['score']:.4f}",
            flush=True,
        )
        return result
