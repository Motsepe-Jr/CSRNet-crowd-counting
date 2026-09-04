"""Utility helpers for CSRNet training and evaluation."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch


def select_device(preferred: str | None = None) -> torch.device:
    """Pick a torch device.

    ``preferred`` may be ``"auto"`` / ``None`` (default behaviour), or any
    explicit string accepted by ``torch.device`` such as ``"cuda"``,
    ``"cuda:0"``, ``"mps"``, or ``"cpu"``. With auto-select we prefer CUDA,
    then Apple MPS, then CPU.
    """
    if preferred and preferred.lower() != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    state: dict,
    is_best: bool,
    task_id: str = "",
    filename: str = "checkpoint.pth.tar",
) -> None:
    """Save a training checkpoint, optionally copying it to ``model_best``.

    ``task_id`` is concatenated with ``filename``; if it contains a directory
    component, the directory is created on demand.
    """
    target = Path(f"{task_id}{filename}")
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, target)
    if is_best:
        best = Path(f"{task_id}model_best.pth.tar")
        shutil.copyfile(target, best)


class AverageMeter:
    """Computes and stores the average and current value."""

    __slots__ = ("val", "avg", "sum", "count")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0
