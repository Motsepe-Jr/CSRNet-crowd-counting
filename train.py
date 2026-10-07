"""CSRNet training script.

Modernised for Python >= 3.10 and PyTorch >= 2.2.

Usage::

    python train.py --train-json part_A_train.json \
                    --val-json   part_A_val.json   \
                    --task partA_

Run ``python train.py --help`` for the full list of flags.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    import wandb
except ImportError:
    wandb = None

import dataset
from image import Augment
from model import CSRNet
from utils import AverageMeter, save_checkpoint, select_device

GT_INTERPOLATIONS = {"cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA}


def autocast(device_type: str, enabled: bool, dtype=None):
    """``torch.amp.autocast`` where available, otherwise a no-op context."""
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "autocast"):
        kwargs = {"device_type": device_type, "enabled": enabled}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return amp_module.autocast(**kwargs)
    return contextlib.nullcontext()


def build_grad_scaler(device_type: str, enabled: bool):
    """AMP scaler that works across torch versions.

    ``torch.amp.GradScaler`` only exists from torch 2.1; older builds have
    ``torch.cuda.amp.GradScaler``. Hardcoding the former makes the script
    unrunnable on torch 2.0, including CPU-only laptops.
    """
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        return amp_module.GradScaler(device_type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"unknown optimizer {args.optimizer!r}")


def lr_at_epoch(epoch: int, args: argparse.Namespace) -> float:
    """Learning rate for ``epoch``, including any warm-up.

    The original code multiplied the LR by ``scales = [1, 1, 1, 1]``, i.e. it
    never decayed at all - a fixed LR dressed up as a schedule. These are real
    schedules.
    """
    base = args.lr

    if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
        # Linear warm-up from 10% of base avoids the large first steps that
        # wreck a pretrained frontend.
        return base * (0.1 + 0.9 * (epoch + 1) / args.warmup_epochs)

    if args.lr_schedule == "none":
        return base

    progress_epoch = epoch - args.warmup_epochs
    total = max(1, args.epochs - args.warmup_epochs)

    if args.lr_schedule == "cosine":
        return args.lr_min + 0.5 * (base - args.lr_min) * (
            1.0 + math.cos(math.pi * min(progress_epoch / total, 1.0))
        )

    if args.lr_schedule == "step":
        decayed = base
        for milestone in args.lr_decay_epochs:
            if epoch >= milestone:
                decayed *= args.lr_decay_factor
        return max(decayed, args.lr_min)

    raise ValueError(f"unknown lr schedule {args.lr_schedule!r}")


def grad_global_norm(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().pow(2).sum())
    return total ** 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch CSRNet")
    parser.add_argument("--train-json", default="part_A_train.json", help="path to train json")
    parser.add_argument("--val-json", default="part_A_val.json", help="path to val/test json")
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="skip validation and checkpoint selection; save the latest checkpoint each epoch",
    )
    parser.add_argument("--pre", "-p", default=None, help="path to a pretrained checkpoint")
    parser.add_argument("--task", default="", help="task id (prefix for checkpoint files)")
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device: 'auto' (default), 'cuda', 'cuda:0', 'mps', 'cpu'",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--print-freq", type=int, default=30)
    parser.add_argument("--seed", type=int, default=int(time.time()) & 0xFFFFFFFF)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="enable torch.amp mixed-precision (CUDA only; no-op on CPU/MPS)",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help=(
            "autocast dtype when --amp is set. bfloat16 has the same exponent "
            "range as float32, so it needs no loss scaling and cannot overflow "
            "the way float16 does; float16 reproduces the older behaviour"
        ),
    )
    parser.add_argument(
        "--gt-downsample",
        choices=tuple(GT_INTERPOLATIONS),
        default="cubic",
        help=(
            "interpolation for shrinking the density map to the 1/8 output "
            "stride. 'cubic' is the original CSRNet recipe; 'area' conserves "
            "the crowd count exactly and suits tight kernels (UCSD, sigma=3)"
        ),
    )
    # ---- optimisation -------------------------------------------------
    parser.add_argument(
        "--optimizer",
        choices=("sgd", "adam", "adamw"),
        default="sgd",
        help="sgd reproduces the paper; adam converges far more reliably here",
    )
    parser.add_argument(
        "--loss-norm",
        choices=("sum", "batch-mean"),
        default="batch-mean",
        help=(
            "how the summed squared error is scaled before backprop. "
            "'sum' is what this repo did: the gradient then grows linearly "
            "with batch size, so a recipe tuned at batch 1 diverges at batch "
            "16. 'batch-mean' divides by 2N, which is the loss the CSRNet "
            "paper actually defines, and makes lr independent of batch size"
        ),
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("none", "step", "cosine"),
        default="cosine",
        help="'none' is the paper's fixed LR; cosine anneals to --lr-min",
    )
    parser.add_argument("--lr-min", type=float, default=0.0, help="floor for the LR schedule")
    parser.add_argument(
        "--lr-decay-epochs",
        type=int,
        nargs="*",
        default=[200, 300],
        help="epochs at which --lr-schedule step multiplies the LR",
    )
    parser.add_argument("--lr-decay-factor", type=float, default=0.1)
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="linearly ramp the LR over this many epochs before the schedule",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=0.0,
        help="clip the global gradient norm to this value (0 disables)",
    )

    # ---- augmentation --------------------------------------------------
    parser.add_argument(
        "--crop-fraction",
        type=float,
        default=0.5,
        help="fraction of each side kept by the random training crop",
    )
    parser.add_argument("--no-hflip", action="store_true", help="disable the random horizontal flip")
    parser.add_argument(
        "--aug-brightness",
        type=float,
        default=0.0,
        help="random brightness jitter, e.g. 0.2 scales pixels by [0.8, 1.2]",
    )
    parser.add_argument(
        "--aug-contrast",
        type=float,
        default=0.0,
        help="random contrast jitter around the patch mean",
    )
    parser.add_argument(
        "--aug-noise",
        type=float,
        default=0.0,
        help="gaussian pixel noise sigma, in 0-255 units",
    )
    parser.add_argument(
        "--min-crop-density",
        type=float,
        default=0.0,
        help=(
            "resample a training crop until it holds at least this many "
            "people (max 8 tries). Useful on ROI-masked data where many "
            "random crops are entirely background"
        ),
    )
    parser.add_argument(
        "--replicate",
        type=int,
        default=4,
        help="how many random crops per image per epoch (original recipe: 4)",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "online", "offline"),
        default="disabled",
        help="anything but 'disabled' also switches W&B on (used by the SLURM scripts)",
    )
    parser.add_argument("--wandb-group", default="", help="W&B run group, for grouping a sweep")
    parser.add_argument("--wandb-run-name", default="", help="W&B run name")
    parser.add_argument("--wandb-tags", default="", help="comma-separated W&B tags")
    parser.add_argument(
        "--wandb-project",
        default="csrnet-crowd-counting",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Weights & Biases entity (team) name; if None, uses your default entity",
    )
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = build_parser().parse_args()

    # Initialize Weights & Biases if enabled
    wandb_run = None
    wandb_enabled = args.wandb or args.wandb_mode != "disabled"
    if wandb_enabled:
        if wandb is None:
            print("WARNING: wandb not installed; skipping W&B logging")
            print("Install with: pip install wandb")
        else:
            print("Initializing Weights & Biases logging...")
            print(f"Project: {args.wandb_project}, Entity: {args.wandb_entity}")
            init_kwargs = {
                "project": args.wandb_project,
                "entity": args.wandb_entity,
                "mode": args.wandb_mode if args.wandb_mode != "disabled" else "online",
            }
            if args.wandb_group:
                init_kwargs["group"] = args.wandb_group
            if args.wandb_run_name:
                init_kwargs["name"] = args.wandb_run_name
            tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
            if tags:
                init_kwargs["tags"] = tags
            wandb_run = wandb.init(
                **init_kwargs,
                config={
                    "learning_rate": args.lr,
                    "momentum": args.momentum,
                    "weight_decay": args.weight_decay,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "device": args.device,
                    "amp": args.amp,
                    "amp_dtype": args.amp_dtype,
                    "gt_downsample": args.gt_downsample,
                    "task": args.task,
                    "no_validation": args.no_validation,
                    "optimizer": args.optimizer,
                    "loss_norm": args.loss_norm,
                    "lr_schedule": args.lr_schedule,
                    "warmup_epochs": args.warmup_epochs,
                    "clip_grad": args.clip_grad,
                    "crop_fraction": args.crop_fraction,
                    "replicate": args.replicate,
                    "aug_brightness": args.aug_brightness,
                    "aug_contrast": args.aug_contrast,
                    "aug_noise": args.aug_noise,
                    "min_crop_density": args.min_crop_density,
                    "train_json": args.train_json,
                },
            )
            print(f"Logging to W&B project '{args.wandb_project}'")

    args.original_lr = args.lr

    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")

    with open(args.train_json, "r") as f:
        train_list = json.load(f)
    if not train_list:
        raise ValueError(f"{args.train_json} does not contain any training images")
    
    val_list = None
    if not args.no_validation:
        with open(args.val_json, "r") as f:
            val_list = json.load(f)

    model = CSRNet().to(device)
    criterion = nn.MSELoss(reduction="sum").to(device)
    optimizer = build_optimizer(model, args)

    use_amp = args.amp and device.type == "cuda"
    # Loss scaling exists to stop float16 underflowing. bfloat16 keeps float32's
    # exponent range, so the scaler is left disabled for it and the AMP branch
    # degrades to a plain backward/step.
    scaler = build_grad_scaler(
        device.type, enabled=use_amp and args.amp_dtype == "float16"
    )

    best_prec1 = float("inf")
    if args.pre:
        if os.path.isfile(args.pre):
            print(f"=> loading checkpoint '{args.pre}'")
            checkpoint = torch.load(args.pre, map_location=device, weights_only=False)
            args.start_epoch = checkpoint.get("epoch", 0)
            best_prec1 = float(checkpoint.get("best_prec1", best_prec1))
            model.load_state_dict(checkpoint["state_dict"])
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"=> loaded checkpoint '{args.pre}' (epoch {args.start_epoch})")
        else:
            print(f"=> no checkpoint found at '{args.pre}'")

    for epoch in range(args.start_epoch, args.epochs):
        args.lr = lr_at_epoch(epoch, args)
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr

        train_loss = train(
            train_list, model, criterion, optimizer, epoch, args, device,
            scaler, use_amp, wandb_run,
        )
        if val_list is not None:
            prec1, rmse = validate(val_list, model, args, device, wandb_run, epoch)
            is_best = prec1 < best_prec1
            best_prec1 = min(prec1, best_prec1)
            print(f" * best MAE {best_prec1:.3f} ")
        else:
            is_best = False
            print(" * validation skipped")
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.pre,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            args.task,
        )

    # Finish W&B run if enabled
    if wandb_run is not None:
        wandb_run.finish()


def train(
    train_list,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    wandb_run=None,
) -> float:
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    losses = AverageMeter()
    per_sample = AverageMeter()
    grad_norms = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = DataLoader(
        dataset.ListDataset(
            train_list,
            shuffle=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            ),
            train=True,
            gt_interpolation=GT_INTERPOLATIONS[args.gt_downsample],
            aug=Augment(
                crop_fraction=args.crop_fraction,
                hflip=not args.no_hflip,
                brightness=args.aug_brightness,
                contrast=args.aug_contrast,
                noise_std=args.aug_noise,
                min_crop_density=args.min_crop_density,
            ),
            replicate=args.replicate,
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    print(
        f"epoch {epoch}, processed {epoch * len(train_loader.dataset)} samples, lr {args.lr:.10f}"
    )

    model.train()
    end = time.time()

    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.to(device, non_blocking=True)
        target = target.float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device.type, use_amp, amp_dtype):
            output = model(img)
            sse = criterion(output, target)
            # The paper's loss is (1/2N) * sum ||pred - gt||^2. Summing without
            # dividing lets the gradient scale with batch size, so an lr tuned
            # at batch 1 takes ~19x larger steps at batch 16 and diverges.
            loss = sse / (2 * img.size(0)) if args.loss_norm == "batch-mean" else sse

        if use_amp:
            scaler.scale(loss).backward()
            # Always unscale before measuring. Reading the norm off the scaled
            # gradients reports the loss-scale factor (tens of thousands), not
            # the real gradient, which makes the number useless for diagnosis.
            scaler.unscale_(optimizer)
            if args.clip_grad > 0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                )
            else:
                grad_norm = grad_global_norm(model)
            scaler.step(optimizer)
            scaler.update()
            # A collapsing scale means fp16 keeps overflowing, and every
            # overflowing step is silently skipped by the scaler - which looks
            # exactly like a model that refuses to learn.
            amp_scale = scaler.get_scale()
        else:
            loss.backward()
            if args.clip_grad > 0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                )
            else:
                grad_norm = grad_global_norm(model)
            optimizer.step()
            amp_scale = float("nan")

        # Track per-sample SSE as well: unlike the optimised loss it is
        # comparable across batch sizes and --loss-norm settings.
        losses.update(loss.item(), img.size(0))
        per_sample.update(sse.item() / img.size(0), img.size(0))
        grad_norms.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"SSE/sample {per_sample.val:.3f} ({per_sample.avg:.3f})\t"
                f"|grad| {grad_norms.val:.2f}\t"
                f"lr {args.lr:.3e}"
            )

            # Log to W&B if enabled
            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch,
                    "batch": epoch * len(train_loader) + i,
                    "train/loss": losses.val,
                    "train/loss_avg": losses.avg,
                    "train/sse_per_sample": per_sample.val,
                    "train/sse_per_sample_avg": per_sample.avg,
                    "train/grad_norm": grad_norms.val,
                    "train/amp_scale": amp_scale,
                    "train/batch_time": batch_time.val,
                    "train/batch_time_avg": batch_time.avg,
                    "train/data_time": data_time.val,
                    "train/data_time_avg": data_time.avg,
                    "learning_rate": args.lr,
                })

    return per_sample.avg


@torch.no_grad()
def validate(
    val_list,
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    wandb_run=None,
    epoch: int = 0,
) -> tuple[float, float]:
    print("begin test")
    test_loader = DataLoader(
        dataset.ListDataset(
            val_list,
            shuffle=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            ),
            train=False,
            gt_interpolation=GT_INTERPOLATIONS[args.gt_downsample],
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model.eval()
    absolute_error = 0.0
    squared_error = 0.0
    sample_count = 0

    for img, target in test_loader:
        img = img.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(img)

        predicted_counts = output.flatten(1).sum(dim=1)
        target_counts = target.flatten(1).sum(dim=1)
        count_error = predicted_counts - target_counts
        absolute_error += count_error.abs().sum().item()
        squared_error += count_error.square().sum().item()
        sample_count += img.size(0)

    if sample_count == 0:
        raise ValueError("validation list is empty")
    mae = absolute_error / sample_count
    rmse = float(np.sqrt(squared_error / sample_count))
    print(f" * MAE {mae:.3f} RMSE {rmse:.3f}")

    # Log to W&B if enabled
    if wandb_run is not None:
        wandb_run.log({
            "epoch": epoch,
            "val/mae": mae,
            "val/rmse": rmse,
        })

    return mae, rmse


if __name__ == "__main__":
    main()