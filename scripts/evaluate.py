from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Running "python scripts/evaluate.py" puts scripts/ on sys.path, not the repo
# root, so dataset/model/utils would not import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dataset
from model import CSRNet
from utils import select_device
from wandb_utils import add_wandb_args, init_wandb


GT_INTERPOLATIONS = {"cubic": cv2.INTER_CUBIC, "area": cv2.INTER_AREA}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a CSR-Net checkpoint on a JSON split.")
    parser.add_argument("--json", required=True, help="path to a JSON file containing image paths")
    parser.add_argument("--checkpoint", required=True, help="checkpoint to evaluate")
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device: 'auto' (default), 'cuda', 'cuda:0', 'mps', 'cpu'",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split-name", default="test", help="human-readable split label")
    parser.add_argument(
        "--gt-downsample",
        choices=tuple(GT_INTERPOLATIONS),
        default="cubic",
        help="must match the setting used at training time",
    )
    add_wandb_args(parser)
    return parser


def build_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


@torch.no_grad()
def evaluate(eval_list: list[str], model: torch.nn.Module, args, device: torch.device) -> float:
    loader = DataLoader(
        dataset.ListDataset(
            eval_list,
            shuffle=False,
            transform=build_transform(),
            train=False,
            gt_interpolation=GT_INTERPOLATIONS[args.gt_downsample],
        ),
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    if len(loader.dataset) == 0:
        raise ValueError("evaluation list is empty")

    model.eval()
    abs_err = 0.0
    n_samples = 0

    for img, target in loader:
        img = img.to(device, non_blocking=True)
        target = target.float().to(device, non_blocking=True)
        output = model(img)

        pred_counts = output.flatten(1).sum(dim=1)
        gt_counts = target.flatten(1).sum(dim=1)
        abs_err += (pred_counts - gt_counts).abs().sum().item()
        n_samples += img.size(0)

    return abs_err / n_samples


def main() -> None:
    args = build_parser().parse_args()
    device = select_device(args.device)
    print(f"Using device: {device}")

    with open(args.json, "r", encoding="utf-8") as handle:
        eval_list = json.load(handle)
    if not eval_list:
        raise ValueError(f"{args.json} does not contain any image paths")

    model = CSRNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])

    wandb_run = init_wandb(
        args,
        config={
            "checkpoint": args.checkpoint,
            "json": args.json,
            "device": str(device),
            "batch_size": args.batch_size,
            "workers": args.workers,
            "split_name": args.split_name,
            "gt_downsample": args.gt_downsample,
        },
        default_project="csr-net",
        job_type="eval",
    )

    mae = evaluate(eval_list, model, args, device)
    print(f" * {args.split_name} MAE {mae:.3f}")

    if wandb_run is not None:
        wandb_run.log({"eval/mae": mae})
        wandb_run.summary["eval_mae"] = mae
        wandb_run.summary["split_name"] = args.split_name
        wandb_run.summary["checkpoint"] = args.checkpoint
        wandb_run.finish()


if __name__ == "__main__":
    main()