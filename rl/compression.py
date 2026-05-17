"""Image compression helpers shared by rewards and metrics."""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import numpy as np
import torch
from PIL import Image


def sample_to_uint8_pil(sample: torch.Tensor) -> Image.Image:
    """Convert a single `[-1, 1]` image tensor to a uint8 PIL image.

    Preserves grayscale samples as mode ``L`` when the tensor has shape
    ``[1, H, W]``. RGB samples with shape ``[3, H, W]`` become mode ``RGB``.
    """
    if sample.ndim != 3:
        raise ValueError(f"expected [C,H,W] sample, got shape {tuple(sample.shape)}")

    c, _h, _w = sample.shape
    arr = (
        sample.detach()
        .clamp(-1, 1)
        .add(1.0)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

    if c == 1:
        return Image.fromarray(arr[0], mode="L")
    if c == 3:
        return Image.fromarray(np.transpose(arr, (1, 2, 0)), mode="RGB")
    raise ValueError(f"expected 1 or 3 channels, got {c}")


def encoded_image_bytes(
    image: Image.Image,
    *,
    format: str,
    quality: Optional[int] = None,
    optimize: bool = False,
    progressive: bool = False,
    subsampling: Optional[int | str] = None,
) -> bytes:
    """Encode a PIL image deterministically and return the encoded bytes."""
    kwargs: dict[str, object] = {"optimize": optimize}
    if format.upper() == "JPEG":
        if image.mode not in {"L", "RGB"}:
            image = image.convert("RGB")
        if quality is not None:
            kwargs["quality"] = quality
        kwargs["progressive"] = progressive
        if image.mode == "RGB" and subsampling is not None:
            kwargs["subsampling"] = subsampling

    with BytesIO() as buf:
        image.save(buf, format=format, **kwargs)
        return buf.getvalue()


def jpeg_bytes_for_sample(
    sample: torch.Tensor,
    *,
    quality: int = 75,
    optimize: bool = False,
    progressive: bool = False,
    subsampling: Optional[int | str] = None,
) -> int:
    """Return deterministic JPEG byte length for one `[-1, 1]` image tensor."""
    image = sample_to_uint8_pil(sample)
    data = encoded_image_bytes(
        image,
        format="JPEG",
        quality=quality,
        optimize=optimize,
        progressive=progressive,
        subsampling=subsampling,
    )
    return len(data)


def jpeg_bpp_for_sample(
    sample: torch.Tensor,
    *,
    quality: int = 75,
    optimize: bool = False,
    progressive: bool = False,
    subsampling: Optional[int | str] = None,
) -> float:
    """Return JPEG bits-per-pixel for one `[-1, 1]` image tensor."""
    if sample.ndim != 3:
        raise ValueError(f"expected [C,H,W] sample, got shape {tuple(sample.shape)}")
    h, w = int(sample.shape[-2]), int(sample.shape[-1])
    nbytes = jpeg_bytes_for_sample(
        sample,
        quality=quality,
        optimize=optimize,
        progressive=progressive,
        subsampling=subsampling,
    )
    return 8.0 * nbytes / float(h * w)


def png_bytes_for_sample(sample: torch.Tensor, *, optimize: bool = False) -> int:
    """Return PNG byte length for one `[-1, 1]` image tensor."""
    image = sample_to_uint8_pil(sample)
    data = encoded_image_bytes(image, format="PNG", optimize=optimize)
    return len(data)
