"""Turn the UCSD pedestrian dataset into the layout CSR-Net already trains on.

The UCSD "peoplecnt" release ships as

  * ``ucsdpeds.zip``   - raw 158x238 greyscale PNG video frames (vidf / vidd)
  * ``vidf-cvpr.zip``  - MATLAB dot annotations, ROI mask and perspective map
                         for the 2000 benchmark frames (vidf1_33_000..009)

Neither is directly trainable. This script converts them into the same
``images/`` + ``ground_truth/*.h5`` structure that ``image.py`` expects for
ShanghaiTech, so no change to ``dataset.py`` / ``train.py`` is needed.

The protocol follows CSRNet (Li et al., CVPR 2018) Sec. 4.3.4 / Table 2:

  * frames are enlarged 238x158 -> 952x632 with bilinear interpolation
  * frames and dot maps are masked with the provided ROI *before* blurring
  * density maps use a fixed Gaussian kernel, sigma = 3
  * frames 601-1400 are the training set, the remaining 1200 are the test set

Example
-------
    python scripts/prepare_ucsd.py \
        --dataset-root UCSD_Crowd_Counting_Dataset \
        --output-dir data_splits
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import scipy.io
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

CLIPS = tuple(range(10))            # vidf1_33_000 .. vidf1_33_009
FRAMES_PER_CLIP = 200               # 10 clips x 200 = the 2000 benchmark frames
ORIG_H, ORIG_W = 158, 238
# ucsdpeds.zip holds ~18k frames (vidf1..vidf6, clips 000-028); only
# vidf1_33_000..009 carry CVPR ground truth, so that is all we look at.
FRAME_RE = re.compile(r"vidf1_33_(00\d)_f(\d{3})\.(?:png|jpg|jpeg)$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# MATLAB helpers
# --------------------------------------------------------------------------- #
def _unwrap(node):
    """Peel the 1x1 object wrappers scipy.io puts around MATLAB struct fields."""
    while isinstance(node, np.ndarray) and node.dtype == object and node.size == 1:
        node = node[0, 0] if node.ndim == 2 else node.flat[0]
    return node


def load_roi_mask(vidf_dir: Path) -> np.ndarray:
    """Return the main-walkway ROI as a ``(158, 238)`` uint8 0/1 mask."""
    mat = scipy.io.loadmat(vidf_dir / "vidf1_33_roi_mainwalkway.mat")
    mask = _unwrap(mat["roi"][0, 0]["mask"]).astype(np.uint8)
    if mask.shape != (ORIG_H, ORIG_W):
        raise ValueError(f"unexpected ROI shape {mask.shape}, expected {(ORIG_H, ORIG_W)}")
    return mask


def load_clip_points(vidf_dir: Path, clip: int) -> list[np.ndarray]:
    """Per-frame ``(N, 2)`` person locations, converted to 0-based pixel coords.

    MATLAB stores pixel centres 1-based, so ``x0 = x_matlab - 0.5`` puts the
    annotation in the numpy convention where pixel ``i`` spans ``[i, i + 1)``.
    That convention reproduces the dataset's own in-ROI counts most closely
    (1649/2000 frames exact, vs 1494/2000 for a naive ``floor``).
    """
    mat = scipy.io.loadmat(vidf_dir / f"vidf1_33_{clip:03d}_frame_full.mat")
    frames = mat["frame"]
    out: list[np.ndarray] = []
    for i in range(frames.shape[1]):
        loc = np.atleast_2d(np.asarray(_unwrap(frames[0, i]["loc"]), dtype=np.float64))
        pts = loc[:, :2] if loc.size else np.zeros((0, 2), dtype=np.float64)
        out.append(pts - 0.5)
    return out


def load_official_roi_counts(vidf_dir: Path, clip: int) -> np.ndarray:
    """The dataset's own in-ROI head count per frame (left-walkers + right)."""
    mat = scipy.io.loadmat(vidf_dir / f"vidf1_33_{clip:03d}_count_roi_mainwalkway.mat")
    counts = mat["count"]
    total = np.zeros(FRAMES_PER_CLIP, dtype=np.int64)
    for i in range(counts.shape[1]):
        total += _unwrap(counts[0, i]).astype(np.int64).ravel()
    return total


# --------------------------------------------------------------------------- #
# Raw data discovery / extraction
# --------------------------------------------------------------------------- #
def maybe_unzip(
    zip_path: Path,
    target: Path,
    marker: Path,
    force: bool,
    only_annotated: bool = False,
) -> None:
    """Extract ``zip_path`` into ``target`` unless ``marker`` already exists.

    ``only_annotated`` keeps just the ten ``vidf1_33_00X`` clips that have
    ground truth - the other ~16000 frames in ``ucsdpeds.zip`` are unusable
    here and cost roughly 700 MB of disk.
    """
    if marker.exists() and not force:
        return
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Download it with:\n"
            f"  curl -o {zip_path} "
            f"http://www.svcl.ucsd.edu/projects/peoplecnt/db/{zip_path.name}"
        )
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as handle:
        members = handle.namelist()
        if only_annotated:
            members = [name for name in members if FRAME_RE.search(name)]
            if not members:
                raise ValueError(f"{zip_path.name} contains no vidf1_33_00X frames")
        print(f"extracting {len(members)} entries from {zip_path.name} -> {target}")
        handle.extractall(target, members=members)


def index_frames(frames_root: Path) -> dict[tuple[int, int], Path]:
    """Map ``(clip, frame_number)`` -> path for every vidf1_33 frame on disk."""
    index: dict[tuple[int, int], Path] = {}
    for path in frames_root.rglob("vidf1_33_*_f*"):
        match = FRAME_RE.search(path.name)
        if match:
            index[(int(match.group(1)), int(match.group(2)))] = path
    return index


# --------------------------------------------------------------------------- #
# Density map generation
# --------------------------------------------------------------------------- #
def points_in_roi(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not len(points):
        return points
    height, width = mask.shape
    xi = np.clip(np.floor(points[:, 0]).astype(int), 0, width - 1)
    yi = np.clip(np.floor(points[:, 1]).astype(int), 0, height - 1)
    return points[mask[yi, xi] > 0]


def density_from_points(
    points: np.ndarray,
    height: int,
    width: int,
    sigma: float,
    edge_mode: str = "reflect",
) -> np.ndarray:
    """Rasterise the dots, then blur once.

    A Gaussian filter is linear, so blurring the summed dot map is identical to
    summing one Gaussian per dot - but it is orders of magnitude faster than the
    per-point loop ShanghaiTech's geometry-adaptive kernel needs.

    ``edge_mode`` decides what happens to people standing on the frame border -
    and the UCSD ROI does touch the left edge. ``"constant"`` (what
    ``scripts/prepare_dataset.py`` uses for ShanghaiTech) lets that Gaussian
    mass fall off the image, which costs ~0.18 people per frame here.
    ``"reflect"`` folds it back in, so the density sums to the dot count.
    """
    dots = np.zeros((height, width), dtype=np.float32)
    if len(points):
        xi = np.clip(np.floor(points[:, 0]).astype(int), 0, width - 1)
        yi = np.clip(np.floor(points[:, 1]).astype(int), 0, height - 1)
        np.add.at(dots, (yi, xi), 1.0)
    if not dots.any():
        return dots
    return gaussian_filter(dots, sigma, mode=edge_mode).astype(np.float32)


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def global_frame_index(clip: int, frame: int) -> int:
    """1-based index over the 2000 concatenated benchmark frames."""
    return clip * FRAMES_PER_CLIP + frame


def assign_split(clip: int, frame: int, scheme: str) -> str:
    """Return 'train' or 'test' for one frame."""
    if scheme == "standard":
        # CSRNet / Chan et al.: frames 601-1400 train, remainder test.
        return "train" if 601 <= global_frame_index(clip, frame) <= 1400 else "test"
    if scheme == "all-train":
        return "train"
    raise ValueError(f"unknown split scheme {scheme!r}")


def carve_validation(
    train_records: list[dict],
    val_ratio: float,
    mode: str,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Split the training records into ``(train, val)``.

    ``contiguous`` (the default) holds out the tail of each training clip.
    UCSD frames are consecutive video at 10 fps, so a random hold-out puts
    near-duplicate frames on both sides of the split and reports an
    optimistically low validation MAE.
    """
    if val_ratio <= 0.0:
        return train_records, []
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"--val-ratio must be in [0, 1), got {val_ratio}")

    if mode == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(train_records))
        n_val = max(1, int(round(len(train_records) * val_ratio)))
        val_idx = set(order[:n_val].tolist())
        train = [r for i, r in enumerate(train_records) if i not in val_idx]
        val = [r for i, r in enumerate(train_records) if i in val_idx]
        return train, val

    if mode != "contiguous":
        raise ValueError(f"unknown --val-mode {mode!r}")

    by_clip: dict[int, list[dict]] = {}
    for record in train_records:
        by_clip.setdefault(record["clip"], []).append(record)

    train: list[dict] = []
    val: list[dict] = []
    for clip in sorted(by_clip):
        clip_records = sorted(by_clip[clip], key=lambda r: r["frame"])
        n_val = max(1, int(round(len(clip_records) * val_ratio)))
        n_val = min(n_val, len(clip_records) - 1)
        train.extend(clip_records[:-n_val])
        val.extend(clip_records[-n_val:])
    return train, val


def write_json(path: Path, values: list[str], indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=indent) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert the UCSD pedestrian dataset into CSR-Net training format."
    )
    parser.add_argument(
        "--dataset-root",
        default="UCSD_Crowd_Counting_Dataset",
        help="folder holding raw/ucsdpeds.zip and raw/vidf-cvpr.zip (or their extracted trees)",
    )
    parser.add_argument(
        "--output-dir",
        default="data_splits",
        help="directory where the JSON split files are written",
    )
    parser.add_argument(
        "--processed-name",
        default="ucsd_processed",
        help="subfolder of --dataset-root that receives train_data/ and test_data/",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="upsampling factor; 4 reproduces CSRNet's 952x632 frames",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="fixed Gaussian sigma, in *output* pixels (CSRNet Table 2 uses 3)",
    )
    parser.add_argument(
        "--roi",
        choices=("mask", "none"),
        default="mask",
        help="'mask' keeps only in-ROI people and blacks out the rest of the frame",
    )
    parser.add_argument(
        "--edge-mode",
        choices=("reflect", "constant"),
        default="reflect",
        help=(
            "Gaussian boundary handling. 'reflect' keeps sum(density) == in-ROI "
            "head count; 'constant' matches the ShanghaiTech script and loses "
            "~0.18 people per frame at the image border"
        ),
    )
    parser.add_argument(
        "--mask-density",
        action="store_true",
        help="also zero the blurred density outside the ROI (loses a little count mass)",
    )
    parser.add_argument(
        "--split",
        choices=("standard", "all-train"),
        default="standard",
        help="'standard' = frames 601-1400 train / rest test (the CSRNet protocol)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="validation share carved out of the train split",
    )
    parser.add_argument(
        "--val-mode",
        choices=("contiguous", "random"),
        default="contiguous",
        help="'contiguous' holds out the tail of each training clip (recommended for video)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="ucsd", help="stem for the JSON split filenames")
    parser.add_argument(
        "--image-format",
        choices=("png", "jpg"),
        default="png",
        help="png keeps the frames lossless; jpg is ~3x smaller on disk",
    )
    parser.add_argument(
        "--compression",
        type=int,
        default=4,
        help="gzip level for the .h5 density maps (0 disables); they are mostly zeros",
    )
    parser.add_argument("--json-indent", type=int, default=2)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only process the first N frames of each clip (smoke test)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild frames / density maps that already exist",
    )
    parser.add_argument(
        "--extract-all-frames",
        action="store_true",
        help="unzip every ucsdpeds clip, not just the ten annotated vidf1_33_00X ones",
    )
    parser.add_argument(
        "--preview",
        default="",
        help="optional path for a PNG contact sheet of a few frames + density maps",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.dataset_root).resolve()
    out_json_dir = Path(args.output_dir).resolve()
    processed = root / args.processed_name

    # ----------------------------------------------------------------- inputs
    vidf_dir = root / "vidf-cvpr"
    maybe_unzip(
        root / "raw" / "vidf-cvpr.zip",
        root,
        vidf_dir / "vidf1_33_roi_mainwalkway.mat",
        args.force,
    )
    frames_root = root / "ucsdpeds"
    maybe_unzip(
        root / "raw" / "ucsdpeds.zip",
        root,
        frames_root,
        False,
        only_annotated=not args.extract_all_frames,
    )

    mask = load_roi_mask(vidf_dir)
    frame_index = index_frames(frames_root if frames_root.exists() else root)
    if not frame_index:
        raise FileNotFoundError(
            f"no vidf1_33_*_f*.png frames found under {frames_root}. "
            "Did ucsdpeds.zip finish downloading and extracting?"
        )
    print(f"found {len(frame_index)} raw frames; ROI covers {mask.mean() * 100:.1f}% of each frame")

    # The annotation coordinates are only meaningful against the native size.
    with Image.open(next(iter(frame_index.values()))) as probe:
        if probe.size != (ORIG_W, ORIG_H):
            raise ValueError(
                f"expected {ORIG_W}x{ORIG_H} source frames, got {probe.size[0]}x{probe.size[1]}"
            )

    out_h, out_w = ORIG_H * args.scale, ORIG_W * args.scale
    mask_up = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    print(
        f"output resolution {out_w}x{out_h} (WxH), sigma={args.sigma}, "
        f"roi={args.roi}, edge-mode={args.edge_mode}"
    )

    # ------------------------------------------------------------- conversion
    n_frames = min(args.limit, FRAMES_PER_CLIP) if args.limit else FRAMES_PER_CLIP
    records: list[dict] = []
    count_gap: list[float] = []
    mass_gap: list[float] = []

    for clip in CLIPS:
        clip_points = load_clip_points(vidf_dir, clip)
        official = load_official_roi_counts(vidf_dir, clip)

        for frame in tqdm(range(1, n_frames + 1), desc=f"vidf1_33_{clip:03d}", unit="frame"):
            src = frame_index.get((clip, frame))
            if src is None:
                raise FileNotFoundError(f"missing raw frame vidf1_33_{clip:03d}_f{frame:03d}")

            split = assign_split(clip, frame, args.split)
            stem = f"vidf1_33_{clip:03d}_f{frame:03d}"
            img_path = processed / f"{split}_data" / "images" / f"{stem}.{args.image_format}"
            h5_path = processed / f"{split}_data" / "ground_truth" / f"{stem}.h5"

            points = clip_points[frame - 1]
            if args.roi == "mask":
                points = points_in_roi(points, mask)
            count = len(points)

            if args.force or not (img_path.exists() and h5_path.exists()):
                img_path.parent.mkdir(parents=True, exist_ok=True)
                h5_path.parent.mkdir(parents=True, exist_ok=True)

                with Image.open(src) as raw:
                    resized = raw.convert("L").resize((out_w, out_h), Image.BILINEAR)
                arr = np.asarray(resized)
                if args.roi == "mask":
                    arr = arr * mask_up
                Image.fromarray(arr).save(img_path)

                density = density_from_points(
                    points * args.scale, out_h, out_w, args.sigma, args.edge_mode
                )
                if args.mask_density:
                    density = density * mask_up
                h5_kwargs = (
                    {"compression": "gzip", "compression_opts": args.compression}
                    if args.compression
                    else {}
                )
                with h5py.File(h5_path, "w") as handle:
                    handle.create_dataset("density", data=density, **h5_kwargs)
                    handle.attrs["count"] = count
                    handle.attrs["official_roi_count"] = int(official[frame - 1])
                    handle.attrs["clip"] = clip
                    handle.attrs["frame"] = frame
                    handle.attrs["sigma"] = args.sigma
                    handle.attrs["scale"] = args.scale
                    handle.attrs["edge_mode"] = args.edge_mode
                mass_gap.append(abs(float(density.sum()) - count))

            count_gap.append(count - int(official[frame - 1]))
            records.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "global": global_frame_index(clip, frame),
                    "split": split,
                    "subset": split,
                    "count": count,
                    "official_roi_count": int(official[frame - 1]),
                    "image": str(img_path),
                }
            )

    # ----------------------------------------------------------------- splits
    train_records = [r for r in records if r["split"] == "train"]
    test_records = [r for r in records if r["split"] == "test"]
    train_only, val_only = carve_validation(train_records, args.val_ratio, args.val_mode, args.seed)
    if not val_only:                      # train.py always needs a non-empty val list
        val_only = train_only[-1:]

    val_images = {r["image"] for r in val_only}
    for record in records:
        if record["image"] in val_images:
            record["subset"] = "val"

    prefix = args.prefix
    write_json(out_json_dir / f"{prefix}_train.json", [r["image"] for r in train_only], args.json_indent)
    write_json(out_json_dir / f"{prefix}_val.json", [r["image"] for r in val_only], args.json_indent)
    write_json(out_json_dir / f"{prefix}_test.json", [r["image"] for r in test_records], args.json_indent)
    write_json(
        out_json_dir / f"{prefix}_train_full.json",
        [r["image"] for r in train_records],
        args.json_indent,
    )
    write_json(
        out_json_dir / f"{prefix}_train_with_val.json",
        [r["image"] for r in train_records],
        args.json_indent,
    )

    manifest = out_json_dir / f"{prefix}_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip",
                "frame",
                "global",
                "split",
                "subset",
                "count",
                "official_roi_count",
                "image",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    # ---------------------------------------------------------------- summary
    counts = np.array([r["count"] for r in records], dtype=float)
    gap = np.array(count_gap, dtype=float)
    print()
    print(f"frames processed : {len(records)}")
    print(f"  train          : {len(train_only)}")
    print(f"  val            : {len(val_only)}  ({args.val_mode})")
    print(f"  test           : {len(test_records)}")
    print(f"people per frame : mean {counts.mean():.2f}  min {counts.min():.0f}  max {counts.max():.0f}")
    print(f"vs official ROI  : mean signed {gap.mean():+.3f}, mean abs {np.abs(gap).mean():.3f}")
    if mass_gap:
        print(f"density sum error: {np.mean(mass_gap):.4f} people (edge-mode={args.edge_mode})")
    print(f"images + .h5     : {processed}")
    print(f"split JSONs      : {out_json_dir}\\{prefix}_*.json")
    print(f"manifest         : {manifest}")

    if args.preview:
        make_preview(records, Path(args.preview))


def make_preview(records: list[dict], out_path: Path) -> None:
    """Save a contact sheet of a few processed frames next to their density maps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = [records[i] for i in np.linspace(0, len(records) - 1, 4).astype(int)]
    fig, axes = plt.subplots(2, len(picks), figsize=(4 * len(picks), 5.5))
    for col, record in enumerate(picks):
        image_path = Path(record["image"])
        img = np.asarray(Image.open(image_path))
        gt_path = image_path.parent.parent / "ground_truth" / f"{image_path.stem}.h5"
        with h5py.File(gt_path, "r") as handle:
            density = np.asarray(handle["density"])
        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(
            f"{image_path.stem}\n{record['split']} | dots = {record['count']}", fontsize=8
        )
        axes[1, col].imshow(density, cmap="jet")
        axes[1, col].set_title(f"density sum = {density.sum():.2f}", fontsize=8)
        for row in (0, 1):
            axes[row, col].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    print(f"preview          : {out_path}")


if __name__ == "__main__":
    main()
