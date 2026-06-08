from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_image(image_source: bytes | Path, image_size: int) -> Image.Image:
    try:
        if isinstance(image_source, bytes):
            image = Image.open(BytesIO(image_source))
        else:
            image = Image.open(image_source)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    return image.convert("RGB").resize((image_size, image_size))


def image_to_features(
    image_source: bytes | Path,
    *,
    image_size: int = 224,
    histogram_bins: int = 24,
) -> np.ndarray:
    image = load_image(image_source, image_size)
    pixels = np.asarray(image, dtype=np.float32) / 255.0

    channel_histograms = [
        np.histogram(pixels[:, :, channel], bins=histogram_bins, range=(0.0, 1.0), density=True)[0]
        for channel in range(3)
    ]
    color_mean = pixels.mean(axis=(0, 1))
    color_std = pixels.std(axis=(0, 1))

    return np.concatenate([*channel_histograms, color_mean, color_std]).astype(np.float32)


def image_metadata(image_source: bytes | Path, *, image_size: int = 224) -> dict:
    image = load_image(image_source, image_size)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    mean_rgb = pixels.mean(axis=(0, 1))

    return {
        "width": image.width,
        "height": image.height,
        "mean_rgb": {
            "red": round(float(mean_rgb[0]), 4),
            "green": round(float(mean_rgb[1]), 4),
            "blue": round(float(mean_rgb[2]), 4),
        },
    }


def iter_labeled_images(data_dir: Path) -> list[tuple[Path, str]]:
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    samples: list[tuple[Path, str]] = []
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_SUFFIXES:
                samples.append((image_path, class_dir.name))

    if not samples:
        raise ValueError(
            "No training images found. Expected data/raw/<label>/*.jpg or similar."
        )

    return samples
