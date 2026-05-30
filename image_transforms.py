"""Image transform helpers for cache building and preprocessing."""


def _convert_to_rgb(image):
    return image.convert("RGB")


def _scale_to_minus1_1(x):
    return x * 2 - 1


def build_cache_transform(
    image_size: int = 256,
    crop: str = "resize",
    hflip: bool = False,
):
    """Build the deterministic image transform used before VAE encoding.

    The current Stage 2 source is the Kaggle ImageNet256 mirror. The cache policy
    assumes images should be converted to RGB and resized directly to 256 by 256.
    """
    from torchvision import transforms

    if crop != "resize":
        raise ValueError(f"Unknown VAE cache crop policy: {crop!r}")
    ops = [
        transforms.Lambda(_convert_to_rgb),
        transforms.Resize((image_size, image_size)),
    ]
    if hflip:
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Lambda(_scale_to_minus1_1),
        ]
    )
    return transforms.Compose(ops)
