from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import scipy.spatial
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


PART_LAYOUTS = {
    "A": {
        "dataset_dir": "part_A_final",
        "json_prefix": "part_A",
        "fixed_sigma": None,
    },
    "B": {
        "dataset_dir": "part_B_final",
        "json_prefix": "part_B",
        "fixed_sigma": 15.0,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate ShanghaiTech density maps and CSR-Net JSON splits."
    )
    parser.add_argument(
        "--dataset-root",
        default="ShanghaiTech_Crowd_Counting_Dataset",
        help="root folder that contains part_A_final/ and part_B_final/",
    )
    parser.add_argument(
        "--output-dir",
        default="data_splits",
        help="directory where JSON split files will be written",
    )
    parser.add_argument(
        "--part",
        choices=("A", "B", "both"),
        default="both",
        help="which ShanghaiTech partition to prepare",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="fraction of train_data images to hold out for validation",
    )
    parser.add_argument("--seed", type=int, default=42, help="split RNG seed")
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="optional cap for train_data images, useful for smoke tests",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="optional cap for test_data images, useful for smoke tests",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenerate .h5 density maps even when they already exist",
    )
    parser.add_argument(
        "--json-indent",
        type=int,
        default=2,
        help="indent level for output JSON files",
    )
    return parser


def _ground_truth_mat_path(image_path: Path) -> Path:
    return image_path.parent.parent / "ground_truth" / f"GT_{image_path.stem}.mat"


def _ground_truth_h5_path(image_path: Path) -> Path:
    return image_path.parent.parent / "ground_truth" / f"{image_path.stem}.h5"


def _find_points(node) -> np.ndarray | None:
    if isinstance(node, dict):
        for value in node.values():
            points = _find_points(value)
            if points is not None:
                return points
        return None

    if not isinstance(node, np.ndarray):
        return None

    if node.dtype == object:
        for value in node.flat:
            points = _find_points(value)
            if points is not None:
                return points
        return None

    if np.issubdtype(node.dtype, np.number) and node.ndim == 2 and node.shape[1] == 2:
        return np.asarray(node, dtype=np.float32)

    return None


def load_annotation_points(mat_path: Path) -> np.ndarray:
    data = scipy.io.loadmat(mat_path)

    if "annPoints" in data:
        points = np.asarray(data["annPoints"], dtype=np.float32)
        if points.ndim == 2 and points.shape[1] == 2:
            return points

    if "image_info" in data:
        try:
            return np.asarray(data["image_info"][0, 0][0, 0][0], dtype=np.float32)
        except Exception:
            points = _find_points(data["image_info"])
            if points is not None:
                return points

    points = _find_points(data)
    if points is None:
        raise ValueError(f"Could not extract crowd annotations from {mat_path}")
    return np.asarray(points, dtype=np.float32)


def rasterise_points(points: np.ndarray, height: int, width: int) -> np.ndarray:
    gt = np.zeros((height, width), dtype=np.float32)
    for x_coord, y_coord in points:
        x_index = int(np.clip(np.floor(x_coord), 0, width - 1))
        y_index = int(np.clip(np.floor(y_coord), 0, height - 1))
        gt[y_index, x_index] += 1.0
    return gt


def gaussian_filter_density(gt: np.ndarray) -> np.ndarray:
    density = np.zeros(gt.shape, dtype=np.float32)
    non_zero = np.nonzero(gt)
    if len(non_zero[0]) == 0:
        return density

    points = np.column_stack((non_zero[1], non_zero[0])).astype(np.float32)
    tree = scipy.spatial.cKDTree(points.copy(), leafsize=2048)
    k_neighbours = min(4, len(points))
    distances, _ = tree.query(points, k=k_neighbours)
    distances = np.atleast_2d(distances)

    for index, (x_coord, y_coord) in enumerate(points):
        point = np.zeros(gt.shape, dtype=np.float32)
        y_index = int(y_coord)
        x_index = int(x_coord)
        point[y_index, x_index] = gt[y_index, x_index]

        if len(points) > 1:
            sigma = float(np.mean(distances[index][1:]) * 0.1)
        else:
            sigma = float(np.mean(gt.shape) / 4.0)
        sigma = max(sigma, 1.0)
        density += gaussian_filter(point, sigma, mode="constant")

    return density.astype(np.float32)


def fixed_sigma_density(gt: np.ndarray, sigma: float) -> np.ndarray:
    if not np.any(gt):
        return np.zeros(gt.shape, dtype=np.float32)
    return gaussian_filter(gt, sigma, mode="constant").astype(np.float32)


def generate_density_map(image_path: Path, fixed_sigma: float | None, force: bool) -> bool:
    h5_path = _ground_truth_h5_path(image_path)
    if h5_path.exists() and not force:
        return False

    points = load_annotation_points(_ground_truth_mat_path(image_path))
    with Image.open(image_path) as image:
        width, height = image.size

    gt = rasterise_points(points, height=height, width=width)
    density = (
        gaussian_filter_density(gt)
        if fixed_sigma is None
        else fixed_sigma_density(gt, fixed_sigma)
    )

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as handle:
        handle["density"] = density
    return True


def collect_images(image_dir: Path, limit: int | None) -> list[Path]:
    images = sorted(image_dir.glob("*.jpg"))
    if limit is not None:
        return images[:limit]
    return images


def write_json(path: Path, values: list[str], indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=indent) + "\n", encoding="utf-8")


def create_split_files(
    *,
    prefix: str,
    train_images: list[Path],
    test_images: list[Path],
    output_dir: Path,
    val_ratio: float,
    seed: int,
    indent: int,
) -> dict[str, int]:
    if not train_images:
        raise ValueError(f"No training images found for {prefix}")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    train_full = [str(path.resolve()) for path in train_images]
    test_full = [str(path.resolve()) for path in test_images]

    if len(train_full) == 1:
        write_json(output_dir / f"{prefix}_train.json", train_full, indent)
        write_json(output_dir / f"{prefix}_val.json", train_full, indent)
        write_json(output_dir / f"{prefix}_test.json", test_full, indent)
        write_json(output_dir / f"{prefix}_train_full.json", train_full, indent)
        write_json(output_dir / f"{prefix}_train_with_val.json", train_full, indent)
        return {
            "train": 1,
            "val": 1,
            "test": len(test_full),
            "train_full": 1,
        }

    rng = np.random.default_rng(seed)
    indices = np.arange(len(train_full))
    rng.shuffle(indices)

    if val_ratio <= 0.0:
        val_count = 1
    else:
        val_count = max(1, int(round(len(train_full) * val_ratio)))
        val_count = min(val_count, len(train_full) - 1)

    val_indices = set(indices[:val_count].tolist())
    train_only = [path for idx, path in enumerate(train_full) if idx not in val_indices]
    val_only = [path for idx, path in enumerate(train_full) if idx in val_indices]

    write_json(output_dir / f"{prefix}_train.json", train_only, indent)
    write_json(output_dir / f"{prefix}_val.json", val_only, indent)
    write_json(output_dir / f"{prefix}_test.json", test_full, indent)
    write_json(output_dir / f"{prefix}_train_full.json", train_full, indent)
    write_json(output_dir / f"{prefix}_train_with_val.json", train_full, indent)

    return {
        "train": len(train_only),
        "val": len(val_only),
        "test": len(test_full),
        "train_full": len(train_full),
    }


def prepare_part(
    *,
    part_key: str,
    dataset_root: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    train_limit: int | None,
    test_limit: int | None,
    force: bool,
    indent: int,
) -> None:
    layout = PART_LAYOUTS[part_key]
    base_dir = dataset_root / layout["dataset_dir"]
    train_dir = base_dir / "train_data" / "images"
    test_dir = base_dir / "test_data" / "images"

    train_images = collect_images(train_dir, train_limit)
    test_images = collect_images(test_dir, test_limit)
    if not train_images:
        raise FileNotFoundError(f"No images found in {train_dir}")
    if not test_images:
        raise FileNotFoundError(f"No images found in {test_dir}")

    fixed_sigma = layout["fixed_sigma"]
    generated = 0
    skipped = 0

    for image_path in tqdm(
        train_images + test_images,
        desc=f"Part {part_key} density maps",
        unit="image",
    ):
        if generate_density_map(image_path, fixed_sigma=fixed_sigma, force=force):
            generated += 1
        else:
            skipped += 1

    split_counts = create_split_files(
        prefix=layout["json_prefix"],
        train_images=train_images,
        test_images=test_images,
        output_dir=output_dir,
        val_ratio=val_ratio,
        seed=seed,
        indent=indent,
    )
    print(
        f"Prepared Part {part_key}: generated={generated}, skipped={skipped}, "
        f"train={split_counts['train']}, val={split_counts['val']}, "
        f"test={split_counts['test']}"
    )


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    parts = ("A", "B") if args.part == "both" else (args.part,)

    for part_key in parts:
        prepare_part(
            part_key=part_key,
            dataset_root=dataset_root,
            output_dir=output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
            train_limit=args.train_limit,
            test_limit=args.test_limit,
            force=args.force,
            indent=args.json_indent,
        )


if __name__ == "__main__":
    main()