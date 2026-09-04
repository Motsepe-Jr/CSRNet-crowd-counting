"""Torch ``Dataset`` over a list of image paths for CSRNet."""

from __future__ import annotations

import random
from typing import Callable, Sequence

import cv2
import torch
from torch.utils.data import Dataset

from image import load_data


class ListDataset(Dataset):
    """Dataset that yields ``(image_tensor, density_tensor)`` pairs.

    Parameters
    ----------
    root:
        Sequence of image file paths. Will not be mutated.
    shuffle:
        If True, shuffle the path list once at construction.
    transform:
        Optional torchvision transform applied to the PIL image (typically
        ``ToTensor`` followed by ``Normalize``).
    train:
        When True, replicate the path list 4x (matching the original
        CSRNet sampling) and enable random crop + horizontal flip in
        :func:`image.load_data`.
    gt_interpolation:
        OpenCV interpolation used to downsample the density map to the
        network's 1/8 output stride. ``cv2.INTER_CUBIC`` is the original
        recipe; ``cv2.INTER_AREA`` conserves the crowd count exactly and is
        the better choice for tight fixed-sigma kernels (see
        :func:`image.load_data`).
    """

    def __init__(
        self,
        root: Sequence[str],
        shuffle: bool = True,
        transform: Callable | None = None,
        train: bool = False,
        gt_interpolation: int = cv2.INTER_CUBIC,
    ) -> None:
        paths = list(root) * 4 if train else list(root)
        if shuffle:
            random.shuffle(paths)

        self.lines = paths
        self.transform = transform
        self.train = train
        self.gt_interpolation = gt_interpolation

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self.lines):
            raise IndexError(index)

        img_path = self.lines[index]
        img, target = load_data(img_path, self.train, self.gt_interpolation)

        if self.transform is not None:
            img = self.transform(img)

        target_tensor = torch.from_numpy(target).float()
        return img, target_tensor


# Backwards-compatibility alias for the original lower-camelCase name.
listDataset = ListDataset
