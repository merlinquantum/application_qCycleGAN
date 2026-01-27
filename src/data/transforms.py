"""Shared transform helpers for dataset loading."""

from __future__ import annotations

from torchvision import transforms
from PIL import Image


def build_transform(image_prep: str):
    """
    Construct a torchvision transform pipeline based on a preset name.

    Args:
        image_prep: Identifier describing the resizing/cropping strategy.

    Returns:
        Callable transform ready to be applied on PIL images.
    """
    if image_prep == "resized_crop_512":
        return transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(512),
            ]
        )
    if image_prep == "resize_286_randomcrop_256x256_hflip":
        return transforms.Compose(
            [
                transforms.Resize((286, 286), interpolation=Image.LANCZOS),
                transforms.RandomCrop((256, 256)),
                transforms.RandomHorizontalFlip(),
            ]
        )
    if image_prep in {"resize_256", "resize_256x256"}:
        return transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=Image.LANCZOS),
            ]
        )
    if image_prep in {"resize_512", "resize_512x512"}:
        return transforms.Compose(
            [
                transforms.Resize((512, 512), interpolation=Image.LANCZOS),
            ]
        )
    if image_prep in {"resize_128", "resize_128x128"}:
        return transforms.Compose(
            [
                transforms.Resize((128, 128), interpolation=Image.LANCZOS),
            ]
        )
    if image_prep in {"resize_64", "resize_64x64"}:
        return transforms.Compose(
            [
                transforms.Resize((64, 64), interpolation=Image.LANCZOS),
            ]
        )
    if image_prep == "no_resize":
        return transforms.Lambda(lambda x: x)
    raise ValueError(f"Unknown image_prep preset: {image_prep}")
