"""Sample loading and on-the-fly augmentation for CSRNet."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image


def _ground_truth_path(img_path: str | Path) -> Path:
    """Return the ``.h5`` ground-truth path for an image path.

    Mirrors the layout used by ``make_dataset.ipynb``: replace the
    ``images`` directory with ``ground_truth`` and the ``.jpg`` extension
    with ``.h5``.
    """
    p = Path(img_path)
    parts = list(p.parts)
    # Replace the *last* occurrence of "images" so paths like
    # /.../part_A/images/IMG_1.jpg become /.../part_A/ground_truth/IMG_1.h5.
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "ground_truth"
            break
    return Path(*parts).with_suffix(".h5")


def load_data(
    img_path: str | Path,
    train: bool = True,
    interpolation: int = cv2.INTER_CUBIC,
):
    """Load an RGB image and its density map ground truth.

    The density map is downsampled by 8 (matching the CSRNet output stride)
    and multiplied by ``64`` so the integral (i.e. crowd count) is
    preserved.

    ``interpolation`` selects how that downsample is done. ``INTER_CUBIC``
    (the default) matches the original CSRNet recipe, but it *samples* the
    density rather than integrating it: with narrow Gaussians the count of a
    single map wanders by a couple of percent depending on where the blobs
    land relative to the 8x8 grid. ``cv2.INTER_AREA`` averages each 8x8 block
    instead, so ``x64`` restores the count exactly - worth using for
    fixed-sigma datasets with tight kernels such as UCSD (sigma = 3).

    When ``train`` is True, applies a random half-size crop and a 50% chance
    horizontal flip, as in the original CSRNet recipe.
    """
    img_path = Path(img_path)
    gt_path = _ground_truth_path(img_path)

    img = Image.open(img_path).convert("RGB")
    with h5py.File(gt_path, "r") as gt_file:
        target = np.asarray(gt_file["density"], dtype=np.float32)

    if train:
        crop_w, crop_h = img.size[0] // 2, img.size[1] // 2
        dx = int(random.random() * (img.size[0] - crop_w))
        dy = int(random.random() * (img.size[1] - crop_h))

        img = img.crop((dx, dy, dx + crop_w, dy + crop_h))
        target = target[dy : dy + crop_h, dx : dx + crop_w]

        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            target = np.fliplr(target).copy()

    new_w = target.shape[1] // 8
    new_h = target.shape[0] // 8
    target = cv2.resize(target, (new_w, new_h), interpolation=interpolation) * 64.0

    return img, target.astype(np.float32)
