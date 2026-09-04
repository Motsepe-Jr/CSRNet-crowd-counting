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

import dataset
from model import CSRNet
from utils import AverageMeter, save_checkpoint, select_device
from wandb_utils import add_wandb_args, init_wandb


GT_INTERPOLATIONS = {"cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA}


def build_grad_scaler(device_type: str, enabled: bool):
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        return amp_module.GradScaler(device_type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch CSRNet")
    parser.add_argument("--train-json", default="part_A_train.json", help="path to train json")
    parser.add_argument("--val-json", default="part_A_val.json", help="path to val/test json")
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
    add_wandb_args(parser)
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = build_parser().parse_args()

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
    with open(args.val_json, "r") as f:
        val_list = json.load(f)
    if not train_list:
        raise ValueError(f"{args.train_json} does not contain any training images")
    if not val_list:
        raise ValueError(f"{args.val_json} does not contain any validation images")
    print(f"Loaded {len(train_list)} training images and {len(val_list)} validation images")

    model = CSRNet().to(device)
    criterion = nn.MSELoss(reduction="sum").to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    use_amp = args.amp and device.type == "cuda"
    scaler = build_grad_scaler(device.type, enabled=use_amp)
    wandb_run = init_wandb(
        args,
        config={
            "train_json": args.train_json,
            "val_json": args.val_json,
            "pretrained": args.pre,
            "task": args.task,
            "device": str(device),
            "epochs": args.epochs,
            "start_epoch": args.start_epoch,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "print_freq": args.print_freq,
            "seed": args.seed,
            "amp": use_amp,
            "gt_downsample": args.gt_downsample,
            "train_size": len(train_list),
            "val_size": len(val_list),
        },
        default_project="csr-net",
        job_type="train",
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
        adjust_learning_rate(optimizer, epoch, args)

        train_loss = train(
            train_list,
            model,
            criterion,
            optimizer,
            epoch,
            args,
            device,
            scaler,
            use_amp,
        )
        prec1 = validate(val_list, model, args, device)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "lr": args.lr,
                    "train/loss": train_loss,
                    "val/mae": prec1,
                    "val/best_mae": best_prec1,
                }
            )
        print(f" * best MAE {best_prec1:.3f} ")
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

    if wandb_run is not None:
        wandb_run.summary["best_val_mae"] = best_prec1
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
) -> float:
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
        # Density map -> (B, 1, H, W) float 500, 200,


        target = target.unsqueeze(1).float().to(device, non_blocking=True)

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

    return losses.avg


@torch.no_grad()
def validate(
    val_list,
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
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
    abs_err = 0.0
    n_samples = 0

    for img, target in test_loader:
        img = img.to(device, non_blocking=True)
        target = target.float().to(device, non_blocking=True)
        output = model(img)

        # Sum of predicted density - sum of GT density per sample.
        pred_counts = output.flatten(1).sum(dim=1)
        gt_counts = target.flatten(1).sum(dim=1)
        abs_err += (pred_counts - gt_counts).abs().sum().item()
        n_samples += img.size(0)

    mae = abs_err / max(n_samples, 1)
    print(f" * MAE {mae:.3f} ")
    return mae


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
