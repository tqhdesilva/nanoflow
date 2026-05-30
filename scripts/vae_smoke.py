"""Smoke test a configured VAE on a small folder of images."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.utils import save_image  # noqa: E402

import config as _config  # noqa: E402,F401


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _find_images(root: Path, limit: int) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    paths = [p for p in root.rglob("*") if p.suffix.lower() in suffixes]
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"No images found under {root}")
    return paths[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-root",
        default="/tmp/data/imagenet256-test/ImageNet",
        help="Folder containing a tiny ImageNet-like sample",
    )
    parser.add_argument("--vae", default="sd_vae_ft_ema", help="Hydra VAE config name")
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", default="/tmp/nanoflow_vae_smoke.png")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    config_dir = os.path.abspath("configs")
    overrides = [
        f"vae={args.vae}",
        "vae_transform=imagenet256_resize",
        f"device={args.device}",
    ]
    if args.local_files_only:
        overrides.append("vae.local_files_only=true")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    vae = hydra.utils.instantiate(cfg.vae, device=args.device)
    transform = hydra.utils.instantiate(cfg.vae_transform)
    image_paths = _find_images(Path(args.image_root), args.batch_size)
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(transform(image))
    x = torch.stack(images)
    z = vae.encode(x)
    y = vae.decode(z).cpu()

    grid = torch.cat([x.cpu(), y], dim=0).clamp(-1, 1) * 0.5 + 0.5
    save_image(grid, args.output, nrow=len(images))
    print(f"input_shape={tuple(x.shape)} latent_shape={tuple(z.shape)}")
    print(f"decoded_shape={tuple(y.shape)} output={args.output}")


if __name__ == "__main__":
    main()
