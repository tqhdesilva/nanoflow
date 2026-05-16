"""Generation metrics, polymorphic and instantiated via Hydra `_target_`.

Each metric is a callable taking a `(N, C, H, W)` tensor in `[-1, 1]` and
returning a result dict that `main.py` aggregates into the run dir.
"""

import tempfile
from pathlib import Path

import torch
from torchvision.utils import save_image


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
