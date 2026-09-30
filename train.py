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
import json
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
from model import CSRNet
from utils import AverageMeter, save_checkpoint, select_device

GT_INTERPOLATIONS = {"cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA}


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
        "--gt-downsample",
        choices=tuple(GT_INTERPOLATIONS),
        default="cubic",
        help=(
            "interpolation for shrinking the density map to the 1/8 output "
            "stride. 'cubic' is the original CSRNet recipe; 'area' conserves "
            "the crowd count exactly and suits tight kernels (UCSD, sigma=3)"
        ),
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="enable Weights & Biases logging",
    )
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
    if args.wandb:
        if wandb is None:
            print("WARNING: wandb not installed; skipping W&B logging")
            print("Install with: pip install wandb")
        else:
            print("Initializing Weights & Biases logging...")
            print(f"Project: {args.wandb_project}, Entity: {args.wandb_entity}")
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config={
                    "learning_rate": args.lr,
                    "momentum": args.momentum,
                    "weight_decay": args.weight_decay,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "device": args.device,
                    "amp": args.amp,
                    "gt_downsample": args.gt_downsample,
                    "task": args.task,
                    "no_validation": args.no_validation,
                },
            )
            print(f"Logging to W&B project '{args.wandb_project}'")

    # Step schedule used by adjust_learning_rate (kept identical to the
    # original recipe).
    args.original_lr = args.lr
    args.steps = [-1, 1, 100, 150]
    args.scales = [1, 1, 1, 1]

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
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
        adjust_learning_rate(optimizer, epoch, args)

        train(train_list, model, criterion, optimizer, epoch, args, device, scaler, use_amp, wandb_run)
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
) -> None:
    losses = AverageMeter()
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
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(img) # H, W, 1, B
            loss = criterion(output, target)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), img.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
            )

            # Log to W&B if enabled
            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch,
                    "batch": epoch * len(train_loader) + i,
                    "train/loss": losses.val,
                    "train/loss_avg": losses.avg,
                    "train/batch_time": batch_time.val,
                    "train/batch_time_avg": batch_time.avg,
                    "train/data_time": data_time.val,
                    "train/data_time_avg": data_time.avg,
                    "learning_rate": args.lr,
                })


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


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    """Decay the LR following ``args.steps`` / ``args.scales`` schedule."""
    args.lr = args.original_lr
    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.lr


if __name__ == "__main__":
    main()