# Training CSRNet on the UCSD pedestrian dataset

@Sesethu here we using the same architecture, same training script  a different dataset that needs its
own preprocessing pass. This document covers what the raw UCSD release
contains, what has to happen to it, and how to run the conversion and the
training.

Companion files:

| File | Role |
| --- | --- |
| [`scripts/prepare_ucsd.py`](scripts/prepare_ucsd.py) | raw UCSD → `images/` + `ground_truth/*.h5` + JSON splits |
| [`slurm_scripts/01_bigbatch_prepare_ucsd.sh`](slurm_scripts/01_bigbatch_prepare_ucsd.sh) | cluster job that downloads + converts |
| [`slurm_scripts/10_submit_csrnet_pipeline.sh`](slurm_scripts/10_submit_csrnet_pipeline.sh) | `PART=UCSD` runs the whole chain |

---

## 1. What the raw dataset actually is

The UCSD data comes from
<http://www.svcl.ucsd.edu/projects/peoplecnt/index.htm> as two plain-HTTP
downloads (no HTTPS — `WebFetch`-style tools fail, use `curl`):

| Archive | Size | Contents |
| --- | --- | --- |
| `db/ucsdpeds.zip` | 792 MB | raw greyscale **PNG video frames**, 158×238 px, 10 fps |
| `db/vidf-cvpr.zip` | 2.4 MB | MATLAB **annotations** for the 2000 benchmark frames |
| `db/readme.pdf` | 95 KB | dataset readme |

Inside `vidf-cvpr/`, per clip `vidf1_33_000` … `vidf1_33_009` (10 clips ×
200 frames = **2000 frames**):

| File | What it holds |
| --- | --- |
| `vidf1_33_00X_frame_full.mat` | `frame(i).loc` → `(N, 3)` per-person `(x, y, ·)` dot annotations, plus track `id`, `ldir`, `tdir` |
| `vidf1_33_00X_people_full.mat` | the same annotations grouped per tracked person |
| `vidf1_33_00X_count_roi_mainwalkway.mat` | the dataset's own in-ROI head count per frame, split into left/right walkers |
| `vidf1_33_roi_mainwalkway.mat` | ROI polygon + a `158×238` binary mask (covers **51.9 %** of the frame) |
| `vidf1_33_dmap3.mat` | perspective map (`pmapx`, `pmapy`, `pmapxy`) — how much bigger a person is at each pixel |

Scene statistics after ROI filtering: **11 to 46 people per frame, mean 24.94**
— which matches the "11 to 46" the CSRNet paper quotes, a good sign the
annotation parsing is right.

### The gap to what this repo trains on

`image.py` expects, for every image `…/images/NAME.ext`, a sibling
`…/ground_truth/NAME.h5` holding a float32 `density` dataset **at the image's
own resolution**. ShanghaiTech ships `.mat` dot annotations that
`scripts/prepare_dataset.py` blurs into exactly that. UCSD ships video frames
plus MATLAB structs in a completely different shape, so it needs its own
converter — that is all `scripts/prepare_ucsd.py` is.

---

## 2. What the preprocessing has to do

Seven things, in order. Steps 1–4 follow the CSRNet paper (Li et al., CVPR
2018) §4.3.4 and Table 2, so the numbers stay comparable to the published
**MAE 1.16 / MSE 1.47**.

1. **Upscale 4×.** Frames are 238×158. CSRNet's output stride is 8, so the
   density map would be 29×19 — far too coarse. The paper resizes every frame
   to **952×632** with bilinear interpolation; `--scale 4` does that. Dot
   coordinates are scaled by the same factor, so the count is unchanged.

2. **Apply the ROI.** Only the main walkway is annotated. People outside it
   are dropped from the dot list *before* blurring (the paper's wording), and
   the frame is multiplied by the nearest-neighbour-upscaled mask so the
   network never sees unlabelled pedestrians. `--roi none` disables this.

3. **Convert MATLAB coordinates.** MATLAB puts the centre of pixel 1 at
   `x = 1.0`; numpy puts it at `x = 0.5`. The script uses `x₀ = x_matlab −
   0.5`. That sounds pedantic, but it decides which side of the ROI boundary
   a person falls on: it reproduces the dataset's own in-ROI counts on
   **1649/2000** frames versus **1494/2000** for a naive `floor(x)`.

4. **Blur into a density map.** Fixed Gaussian, **σ = 3** (CSRNet Table 2
   lists UCSD and WorldExpo'10 as fixed-kernel σ = 3; geometry-adaptive
   kernels are only for the dense sets). Because a Gaussian filter is linear,
   the script rasterises all dots and blurs **once** instead of looping per
   person — seconds instead of minutes for 2000 frames.

5. **Keep the count intact at the borders.** The UCSD ROI touches the left
   edge of the frame, so people standing there lose part of their Gaussian off
   the side of the image. With `mode="constant"` (what the ShanghaiTech script
   uses) that costs ~0.18 people per frame — meaningful next to an MAE around
   1.0. Default here is `--edge-mode reflect`, which folds the mass back in
   and makes `sum(density)` equal the head count exactly.

6. **Write the ShanghaiTech layout**, so nothing downstream has to change:

   ```
   UCSD_Crowd_Counting_Dataset/ucsd_processed/
   ├── train_data/
   │   ├── images/vidf1_33_003_f001.png      # 952×632, ROI-masked
   │   └── ground_truth/vidf1_33_003_f001.h5 # float32 density, gzip
   └── test_data/{images,ground_truth}/
   ```

   Each `.h5` also carries `count`, `official_roi_count`, `clip`, `frame`,
   `sigma`, `scale` and `edge_mode` as attributes, for auditing later.

7. **Split by the published protocol.** Frames 601–1400 (clips 003–006) are
   the training set, the other 1200 frames are the test set — the split every
   UCSD paper reports on. Validation is carved out of the training frames.

### Why validation is a *contiguous* hold-out

UCSD is 10 fps video: consecutive frames are near-duplicates. A random
90/10 split puts almost-identical frames on both sides and reports a
validation MAE far better than the model deserves. `--val-mode contiguous`
(the default) instead holds out the **last 10 % of each training clip**, so
all four clips are still represented but the val frames are temporally
separated. `--val-mode random` is there if you want the optimistic number for
comparison.

---

## 3. A change in the training path you should know about

`image.py` shrinks the density map to the network's 1/8 output stride and
multiplies by 64. It did that with `cv2.INTER_CUBIC`, which *samples* the
density rather than integrating it. With ShanghaiTech's wide kernels that is
harmless; with UCSD's σ = 3 the resulting count wanders by **±2.3 % per frame
(observed range 0.96–1.06×)** purely from where the blobs land relative to the
8×8 grid. On ~25 people that is up to ±1.6 people of pure ground-truth noise —
the same order as the entire metric.

So `load_data`, `ListDataset`, `train.py` and `scripts/evaluate.py` now take
an interpolation choice:

| Flag | Behaviour |
| --- | --- |
| `--gt-downsample cubic` | **default**, unchanged original recipe — keep for ShanghaiTech reruns |
| `--gt-downsample area` | averages each 8×8 block, so ×64 restores the count exactly |

Use `area` for UCSD. On one real batch of eight validation frames whose true
counts are 12, 12, 13, 13, 13, 13, 13, 13:

```
cubic → 12.012  12.245  13.572  13.218  12.780  12.512  11.989  13.333
area  → 12.000  12.000  13.000  13.000  13.000  13.000  13.000  13.000
```

That `13.572` is a 4.4 % error on a single frame's target, before the model
has done anything.

**Whatever you pick at training time, pass the same flag to
`scripts/evaluate.py`** — the metric is computed against the downsampled
target, so mixing the two makes the numbers incomparable.

Separately, `scripts/evaluate.py` could not import `dataset` / `model` /
`utils` at all when run as `python scripts/evaluate.py` (Python puts
`scripts/` on `sys.path`, not the repo root). It now inserts the repo root
itself, which also unbreaks `slurm_scripts/03_biggpu_eval.sh`.

---

## 4. Running it

### Download

```bash
mkdir -p UCSD_Crowd_Counting_Dataset/raw
cd UCSD_Crowd_Counting_Dataset/raw
curl -O http://www.svcl.ucsd.edu/projects/peoplecnt/db/ucsdpeds.zip    # 792 MB
curl -O http://www.svcl.ucsd.edu/projects/peoplecnt/db/vidf-cvpr.zip   # 2.4 MB
```

`curl -C -` resumes if the connection drops; the server is slow but supports
range requests.

### Convert

```bash
python scripts/prepare_ucsd.py \
    --dataset-root UCSD_Crowd_Counting_Dataset \
    --output-dir   data_splits \
    --preview      data_splits/ucsd_preview.png
```

The script unzips both archives itself if they are not already extracted, then
writes `data_splits/ucsd_{train,val,test,train_full,train_with_val}.json` plus
`data_splits/ucsd_manifest.csv` (one row per frame: clip, frame, global index,
split, count, official count, path). `--preview` drops a contact sheet of a few
frames next to their density maps — worth eyeballing once.

What a full run produces (this took about five minutes):

```
frames processed : 2000
  train          : 720   (clips 003-006, frames 1-180 of each)
  val            :  80   (clips 003-006, frames 181-200 of each)
  test           : 1200  (clips 000-002 and 007-009)
people per frame : mean 24.94  min 11  max 46
density sum error: 0.0000 people
```

Disk: 281 MB of processed frames + `.h5`, 54 MB of extracted source frames,
758 MB of archives in `raw/` (deletable once converted).

Useful flags:

| Flag | Default | Why change it |
| --- | --- | --- |
| `--scale` | `4` | `1` keeps native 238×158 (much faster, coarser density) |
| `--sigma` | `3.0` | in *output* pixels — see the note below |
| `--roi` | `mask` | `none` trains on the full frame and all annotated people |
| `--mask-density` | off | also zeroes density outside the ROI (loses a little mass) |
| `--val-mode` | `contiguous` | `random` for the optimistic in-video split |
| `--image-format` | `png` | `jpg` is ~3× smaller on disk, lossy |
| `--limit N` | — | only first N frames per clip, for a smoke test |
| `--force` | off | rebuild frames/`.h5` that already exist |

#### A note on σ = 3

The paper gives σ = 3 for UCSD but never says at which resolution — before or
after the 4× enlargement. This script applies it *after*, in the 952×632
frame, which is the literal reading. That makes a tight target: at the
network's 1/8 output stride the effective σ is only 0.38 px. Measured on
`vidf1_33_003_f067` (18 people), with `--gt-downsample area`:

| `--sigma` | σ at output stride | mass in the 18 brightest output pixels | output pixels above 1 % of peak |
| --- | --- | --- | --- |
| 3 (default) | 0.38 px | 44.8 % | 106 |
| 6 | 0.75 px | 21.4 % | 319 |
| 8 | 1.00 px | 14.5 % | 507 |
| 12 | 1.50 px | 9.0 % | 930 |

The count is exact at every σ, so this is a question of how peaked a target
the network has to fit, not of correctness. σ = 3 is the paper-faithful
default and is what to report against MAE 1.16. If training is unstable or
the loss plateaus, `--sigma 8` (a person-sized blob at the output stride) is
the first thing to try — regenerate into a separate tree with
`--processed-name ucsd_processed_s8 --prefix ucsd_s8` so both variants coexist.

### Train

Every UCSD frame is the same size, so unlike ShanghaiTech you can use a real
batch size:

```bash
python train.py \
    --train-json data_splits/ucsd_train.json \
    --val-json   data_splits/ucsd_val.json \
    --task       checkpoints/ucsd/ucsd_ \
    --gt-downsample area \
    --batch-size 16 \
    --lr 1e-6 \
    --epochs 400 \
    --device cuda --amp
```

`--lr 1e-6` is the paper's fixed SGD learning rate; the repo default of `1e-7`
comes from the reference PyTorch port and also works, just slower.

### Evaluate

```bash
python scripts/evaluate.py \
    --json data_splits/ucsd_test.json \
    --checkpoint checkpoints/ucsd/ucsd_model_best.pth.tar \
    --gt-downsample area \
    --split-name ucsd_test
```

### On the cluster

```bash
PART=UCSD sbatch_pipeline=... ./slurm_scripts/10_submit_csrnet_pipeline.sh
```

`PART=UCSD` routes the prep step to `01_bigbatch_prepare_ucsd.sh` (which
downloads the archives on the node if they are missing), names the checkpoints
`ucsd_*`, and defaults `GT_DOWNSAMPLE=area`. `PART=A` / `PART=B` behave exactly
as before. Extra environment knobs: `UCSD_ROOT`, `UCSD_SCALE`, `UCSD_SIGMA`,
`UCSD_ROI`, `UCSD_SPLIT`, `UCSD_IMAGE_FORMAT`, `VAL_MODE`, `DOWNLOAD`.

---

## 5. Things worth knowing before you trust the numbers

* **The dot annotations disagree slightly with the shipped counts.** Counting
  points inside the ROI mask gives, on average, **+0.20 people** more than
  `count_roi_mainwalkway.mat` (exact on 1649/2000 frames, max difference 3).
  The dataset's counts were derived differently. Both are stored in each
  `.h5`; the density map is built from the dots, so the training target is
  self-consistent. If you need to report against the official counts, the
  manifest CSV has both columns.

* **The reported MAE is against the density sum, not a head count.** That is
  how `train.py` and `scripts/evaluate.py` already compute it, and with
  `--gt-downsample area` + `--edge-mode reflect` the two coincide to within
  floating-point noise.

* **CSRNet's UCSD result was not its best-in-class one.** The paper reports
  MAE 1.16 against MCNN's 1.07 — UCSD is sparse, and CSRNet was built for
  dense scenes. Do not expect this dataset to flatter the architecture.

* **The perspective map goes unused.** `vidf1_33_dmap3.mat` gives per-pixel
  person scale (0.78–7.53). A perspective-adaptive σ instead of a fixed 3
  is an obvious experiment, but it departs from the published protocol, so
  the script does not do it by default.

* **`vidd` frames have no annotations.** `ucsdpeds.zip` also contains the
  Peds2 / `vidd` clips; only `vidf1_33_000`–`009` have CVPR ground truth, and
  those are the only frames the script touches.
