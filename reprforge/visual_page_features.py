"""Cheap deterministic image features used by materialization baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def image_features(path: Path, maximum_side: int = 256) -> tuple[float, float, float]:
    if maximum_side <= 0:
        raise ValueError("maximum_side must be positive")
    with Image.open(path) as image:
        grayscale = image.convert("L")
        scale = min(1.0, maximum_side / max(grayscale.size))
        if scale < 1.0:
            grayscale = grayscale.resize(
                tuple(max(1, round(value * scale)) for value in grayscale.size),
                Image.Resampling.BILINEAR,
            )
        values = np.asarray(grayscale, dtype=np.uint8)
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / values.size
    entropy = float(-(probabilities * np.log2(probabilities)).sum() / 8.0)
    horizontal = (
        float(np.abs(np.diff(values.astype(np.float32), axis=1)).mean() / 255.0)
        if values.shape[1] > 1
        else 0.0
    )
    vertical = (
        float(np.abs(np.diff(values.astype(np.float32), axis=0)).mean() / 255.0)
        if values.shape[0] > 1
        else 0.0
    )
    return entropy, 0.5 * (horizontal + vertical), float(np.mean(values < 245))
