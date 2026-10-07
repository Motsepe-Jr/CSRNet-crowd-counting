"""Sample loading and on-the-fly augmentation for CSRNet."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Augment:
    """Training-time augmentation knobs.

    The defaults reproduce the original CSRNet recipe exactly: a random
    half-size crop plus a 50% horizontal flip, nothing else.

    The photometric options (``brightness``, ``contrast``, ``noise_std``)
    only touch pixels, never the density map, so the crowd count is
    unaffected by construction. That matters: anything that rescales the
    image geometrically would also have to rescale the density *values* to
    keep the integral right, which is why no zoom augmentation is offered
    here.
    """

    crop_fraction: float = 0.5      # fraction of each side kept when cropping
    hflip: bool = True
    brightness: float = 0.0         # +/- fraction, e.g. 0.2 -> x[0.8, 1.2]
    contrast: float = 0.0           # +/- fraction around the patch mean
    noise_std: float = 0.0          # gaussian sigma in 0-255 units
    min_crop_density: float = 0.0   # resample a crop holding < this many people

    @property
    def photometric(self) -> bool:
        return bool(self.brightness or self.contrast or self.noise_std)


DEFAULT_AUGMENT = Augment()


def _photometric(img: Image.Image, aug: Augment) -> Image.Image:
    """Brightness / contrast jitter and gaussian noise, in that order."""
    arr = np.asarray(img, dtype=np.float32)

    if aug.brightness:
        arr *= 1.0 + random.uniform(-aug.brightness, aug.brightness)
    if aug.contrast:
        factor = 1.0 + random.uniform(-aug.contrast, aug.contrast)
        mean = arr.mean()
        arr = (arr - mean) * factor + mean
    if aug.noise_std:
        arr += np.random.normal(0.0, aug.noise_std, arr.shape).astype(np.float32)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


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
    aug: Augment | None = None,
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

    When ``train`` is True, applies the augmentation described by ``aug``
    (see :class:`Augment`); the default reproduces the original CSRNet
    recipe of a random half-size crop and a 50% horizontal flip.
    """
    img_path = Path(img_path)
    gt_path = _ground_truth_path(img_path)

    img = Image.open(img_path).convert("RGB")
    with h5py.File(gt_path, "r") as gt_file:
        target = np.asarray(gt_file["density"], dtype=np.float32)

    if train:
        aug = aug or DEFAULT_AUGMENT
        # Round the crop down to a multiple of 8. The density map is later
        # shrunk by 8 and multiplied by 64, which only conserves the count if
        # the reduction really is 8x. A 316x476 crop becomes 39x59, i.e. a
        # factor of 8.10 x 8.07, so x64 under-counts every training target by
        # ~2.1% - while full-frame validation targets stay exact. That silent
        # train/val mismatch puts a floor under the achievable MAE.
        crop_w = (int(img.size[0] * aug.crop_fraction) // 8) * 8
        crop_h = (int(img.size[1] * aug.crop_fraction) // 8) * 8

        # With an ROI-masked dataset a uniformly random crop often lands on
        # blacked-out background holding nobody, which contributes gradient
        # for "predict zero" and nothing else. Retry a few times for a crop
        # that actually contains people before giving up.
        attempts = 8 if aug.min_crop_density > 0 else 1
        for attempt in range(attempts):
            dx = int(random.random() * (img.size[0] - crop_w))
            dy = int(random.random() * (img.size[1] - crop_h))
            window = target[dy : dy + crop_h, dx : dx + crop_w]
            if attempt == attempts - 1 or window.sum() >= aug.min_crop_density:
                break

        img = img.crop((dx, dy, dx + crop_w, dy + crop_h))
        target = window

        if aug.hflip and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            target = np.fliplr(target).copy()

        if aug.photometric:
            img = _photometric(img, aug)

    new_w = target.shape[1] // 8
    new_h = target.shape[0] // 8
    target = cv2.resize(target, (new_w, new_h), interpolation=interpolation) * 64.0

    return img, target.astype(np.float32)
