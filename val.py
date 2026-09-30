"""
Validation script for CSRNet model on ShanghaiTech dataset.
Evaluates model performance by computing Mean Absolute Error (MAE) on test images.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import scipy.io as io
import torch
from matplotlib import cm as CM
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

from model import CSRNet
from utils import select_device


def main():
    """Main validation pipeline."""
    parser = argparse.ArgumentParser(description="Validate a CSRNet model.")
    parser.add_argument("--dataset", default="ShanghaiTech", help="Dataset directory name.")
    parser.add_argument(
        "--checkpoint", default="model_best.pth.tar", help="Checkpoint path, relative to this script or absolute."
    )
    args = parser.parse_args()
    
    # Define data transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    script_dir = Path(__file__).parent
    root = script_dir / "data" / args.dataset

    print("Root directory for dataset:", root)
    
    # Dataset paths
    part_A_train = root / "part_A_final" / "train_data" / "images"
    part_A_test = root / "part_A_final" / "test_data" / "images"
    part_B_train = root / "part_B_final" / "train_data" / "images"
    part_B_test = root / "part_B_final" / "test_data" / "images"

    ucsd_test = root / "ucsd_processed" / "test_data" / "images"

    path_sets = [ucsd_test]
    print("Test image paths:", path_sets)
    
    # Collect image paths
    img_paths: list[str] = []
    for path in path_sets:
        image_paths = [*Path(path).glob("*.jpg"), *Path(path).glob("*.png")]
        img_paths.extend(sorted(str(image_path) for image_path in image_paths))

    print("Collected image paths:", img_paths)
    if not img_paths:
        searched_paths = ", ".join(str(path) for path in path_sets)
        raise FileNotFoundError(
            f"No .jpg or .png test images found in: {searched_paths}"
        )
    
    # Initialize model
    device = select_device("auto")
    print("Using device:", device)
    model = CSRNet()
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = script_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    
    # Evaluate model
    model.eval()
    mae_count = 0.0
    mse_count = 0.0
    mse_pixel = 0.0
    pixel_count = 0
    
    with torch.no_grad():
        for i, img_path in enumerate(img_paths):
            img = transform(Image.open(img_path).convert("RGB")).to(device)

            image_path = Path(img_path)
            gt_h5 = image_path.parent.parent / "ground_truth" / f"{image_path.stem}.h5"
            with h5py.File(gt_h5, "r") as gt_file:
                groundtruth = np.asarray(gt_file["density"])

            output = model(img.unsqueeze(0))
            print(output.shape)
            
            # Count-level metrics
            pred_count = output.detach().cpu().sum().item()
            gt_count = float(np.sum(groundtruth))
            error_count = abs(pred_count - gt_count)
            squared_error_count = error_count ** 2
            
            mae_count += error_count
            mse_count += squared_error_count
            
            # Pixel-level metrics: resize prediction to match ground truth dimensions
            pred_density = output.detach().cpu().squeeze().numpy()
            # Resize prediction to match ground truth size
            pred_density_resized = cv2.resize(
                pred_density, 
                (groundtruth.shape[1], groundtruth.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            pixel_mse = np.mean((pred_density_resized - groundtruth) ** 2)
            mse_pixel += pixel_mse
            pixel_count += 1
            
            print(i, mae_count)

    # Count-level metrics
    final_mae_count = mae_count / len(img_paths)
    final_mse_count = mse_count / len(img_paths)
    final_rmse_count = np.sqrt(final_mse_count)
    
    # Pixel-level metrics
    final_mse_pixel = mse_pixel / pixel_count
    final_rmse_pixel = np.sqrt(final_mse_pixel)
    
    print("\n" + "="*70)
    print("COUNT-LEVEL METRICS (per image)")
    print("="*70)
    print(f"MAE:  {final_mae_count:.4f}")
    print(f"MSE:  {final_mse_count:.4f}")
    print(f"RMSE: {final_rmse_count:.4f}")
    
    print("\n" + "="*70)
    print("PIXEL-LEVEL METRICS (density map)")
    print("="*70)
    print(f"MSE:  {final_mse_pixel:.4f}")
    print(f"RMSE: {final_rmse_pixel:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()