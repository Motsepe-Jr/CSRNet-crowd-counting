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

## 3. The model and the training loop

Nothing about the architecture changes between ShanghaiTech and UCSD — only
the data pipeline feeding it. For anyone reading the code cold:

### The idea

Counting by detection fails once heads occlude each other. So instead of
detecting people, the network regresses a **density map**: a real-valued image
whose *integral is the crowd count*. A person contributes one unit of mass,
spread over a Gaussian blob. Sum the map, get the count.

### The network — [`model.py`](model.py)

| Stage | What it is | Why |
| --- | --- | --- |
| **Frontend** | first 13 conv layers of VGG16, 3 max-pools | pretrained features; the 3 pools are what make the output 1/8 the input size |
| **Backend** | 6 conv layers, all `dilation=2` | dilation widens the receptive field *without* further downsampling, so spatial detail survives |
| **Output** | one `1×1` conv → single channel | the density map, `H/8 × W/8` |

`CSRNet.__init__` initialises everything from a Gaussian (std 0.01), then
overwrites the frontend with the pretrained VGG16 weights — order matters, or
the transfer learning is wiped out. `load_weights=True` skips the VGG download
(useful in tests).

### One training sample — [`image.py`](image.py) → [`dataset.py`](dataset.py)

`ListDataset` holds a list of image paths (from the split JSONs). For each one,
`load_data`:

1. opens the image as RGB and reads the sibling `.h5` density map
2. **if training**: takes a random half-size crop and flips horizontally 50 % of
   the time — and crops the density map by the *same* box, so image and target
   stay aligned
3. shrinks the density to `H/8 × W/8` and multiplies by 64, so the sum (the
   count) survives the resize — this is the step §4 is about

`ListDataset` also replicates the path list 4× when `train=True`, so one
"epoch" draws four differently-cropped views of each frame.

### The loop — [`train.py`](train.py)

```python
output = model(img)                    # (B, 1, H/8, W/8)
loss   = nn.MSELoss(reduction="sum")(output, target)
```

SGD with momentum 0.95, weight decay 5e-4, fixed LR. Every epoch runs
`validate()`, which is where the metric comes from:

```python
mae = mean(|sum(predicted_density) - sum(ground_truth_density)|)
```

That is **MAE on the count**, not on the pixels — the standard crowd-counting
metric. The best-MAE checkpoint is copied to `*_model_best.pth.tar`.

Two details worth knowing: `--amp` enables mixed precision on CUDA only (a
no-op on CPU/MPS), and `--pre` resumes from a checkpoint, restoring epoch,
optimiser state and best score.

---

## 4. A change in the training path you should know about

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

## 5. Running it locally

Useful for inspecting the data and for a short smoke run. Full training wants
the cluster (§6).

### Set up

```bash
git clone https://github.com/Motsepe-Jr/CSRNet-crowd-counting.git
cd CSRNet-crowd-counting

python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate          # macOS / Linux

pip install -r requirements.txt
python -c "import torch, torchvision, cv2, scipy, h5py; print(torch.__version__, torch.cuda.is_available())"
```

If that prints `False` for CUDA you have a CPU-only torch build — fine for
preparing data and reading code, far too slow for 400 epochs.

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

---

## 6. Running it on the cluster

This is the path that actually trains the model — a laptop GPU is optional,
a laptop CPU is not realistic for 400 epochs.

### 6.1 Get the code onto the cluster

```bash
ssh <you>@<cluster-login-node>
cd /datasets/<you>            # or wherever you have quota
git clone https://github.com/Motsepe-Jr/CSRNet-crowd-counting.git
cd CSRNet-crowd-counting
```

> There is also [`scripts/cluster_sync_submit.py`](scripts/cluster_sync_submit.py),
> a paramiko helper that rsync-style uploads the working tree from a laptop and
> submits in one shot. It predates the repo being on GitHub; `git clone` +
> `git pull` is simpler now. If you do use it, set `CSR_CLUSTER_HOST`,
> `CSR_CLUSTER_USER`, `CSR_REMOTE_PROJECT_DIR` and `CSR_CLUSTER_PASSWORD` in
> your environment — it ships with no defaults on purpose.

### 6.2 Submit the whole chain

One command submits four dependent jobs:

```bash
PART=UCSD ./slurm_scripts/10_submit_csrnet_pipeline.sh
```

It prints the job IDs and where the checkpoint will land:

```
setup_job=123456
prep_job=123457
train_job=123458
eval_job=123459
run_stem=csrnet_ucsd_20260911_142233
checkpoint=/…/checkpoints/csrnet_ucsd_20260911_142233/ucsd_model_best.pth.tar
```

Each job runs only if the previous one succeeded (`--dependency=afterok`), so
a failure stops the chain instead of training on half-built data.

| # | Script | Partition | Walltime | What it does |
| --- | --- | --- | --- | --- |
| 00 | `00_bigbatch_setup_venv.sh` | `bigbatch` | 2 h | builds `.venv-cluster/` (or a conda env), installs torch cu121 + `requirements.txt`, **pre-caches the VGG16 weights** so the GPU node never needs internet |
| 01 | `01_bigbatch_prepare_ucsd.sh` | `bigbatch` | 4 h | downloads both archives if absent, then runs `prepare_ucsd.py` |
| 02 | `02_biggpu_train.sh` | `biggpu` | 24 h | `train.py` |
| 03 | `03_biggpu_eval.sh` | `biggpu` | 4 h | `evaluate.py` on the test split using `*_model_best.pth.tar` |

`PART=UCSD` routes step 01 to the UCSD preparer, names checkpoints `ucsd_*`,
and defaults `GT_DOWNSAMPLE=area`. `PART=A` / `PART=B` behave exactly as
before.

### 6.3 Tuning the run

Everything is environment variables — no file edits:

```bash
PART=UCSD \
EPOCHS=400 \
BATCH_SIZE=16 \
LR=1e-6 \
WORKERS=8 \
AMP=true \
WANDB_API_KEY=<key> \
  ./slurm_scripts/10_submit_csrnet_pipeline.sh
```

| Variable | Default | Notes |
| --- | --- | --- |
| `EPOCHS` | `400` | the paper's budget |
| `BATCH_SIZE` | `16` | safe on UCSD — every frame is the same size |
| `LR` | `1e-7` | the paper uses `1e-6`; both converge, `1e-7` is slower |
| `GT_DOWNSAMPLE` | `area` for UCSD | see §4 — don't change it without changing it for eval too |
| `AMP` | `true` | mixed precision, CUDA only |
| `PRECHECKPOINT` | — | resume from a checkpoint |
| `WANDB_API_KEY` | — | if set, `WANDB_MODE` flips to `online` automatically |
| `UCSD_SIGMA` / `UCSD_SCALE` / `UCSD_ROI` / `UCSD_SPLIT` / `VAL_MODE` / `DOWNLOAD` | see §5 | passed through to `prepare_ucsd.py` |

To re-run training without redoing the 2000-frame conversion, skip the chain
and submit the one job:

```bash
PART=UCSD sbatch slurm_scripts/02_biggpu_train.sh
PART=UCSD sbatch slurm_scripts/03_biggpu_eval.sh
```

`PART=UCSD` sets `GT_DOWNSAMPLE=area` on its own in both the chain and these
standalone submissions, so train and eval cannot silently disagree. Setting
`GT_DOWNSAMPLE` yourself still overrides it — just set it the same way for
both.

### 6.4 Watching it

```bash
squeue -u $USER                      # queue state of all four jobs
tail -f slurm-123458.out             # live training log
scancel 123458                       # kill one job
scancel -u $USER                     # kill everything
```

SLURM writes `slurm-<jobid>.out` into the directory you submitted from. The
training log looks like this — `Loss` is the summed MSE per batch, `MAE` is the
count error on the validation split:

```
epoch 0, processed 0 samples, lr 0.0000010000
Epoch: [0][0/180]   Time 0.412 (0.412)  Data 0.098 (0.098)  Loss 13.9047 (13.9047)
begin test
 * MAE 4.729
 * best MAE 4.729
```

### 6.5 Where things land

```
checkpoints/csrnet_ucsd_<timestamp>/
├── ucsd_checkpoint.pth.tar     # last epoch
└── ucsd_model_best.pth.tar     # lowest validation MAE  ← report from this one
```

### 6.6 When it breaks

| Symptom | Cause | Fix |
| --- | --- | --- |
| `prep_job` never starts | `setup_job` failed | read `slurm-<setup id>.out`; usually no Python 3.10–3.12 on the node, so it falls back to conda — check `$HOME/miniconda3` exists |
| Download stalls in step 01 | the UCSD host is slow and plain-HTTP | re-submit; `curl -C -` resumes the partial file |
| `CUDA out of memory` | `BATCH_SIZE` too high for the node | drop to 8 or 4 |
| Eval MAE wildly worse than validation MAE | `GT_DOWNSAMPLE` differed between train and eval | re-run eval with the same value |
| Job killed at 24 h | 400 epochs didn't fit | resume with `PRECHECKPOINT=<path to last checkpoint>` |

### 6.7 What good looks like

The published CSRNet result on UCSD is **MAE 1.16 / MSE 1.47**. Given mean
occupancy is ~25 people per frame, that is roughly 5 % error. An untrained
network sits around MAE 4–5 (it predicts near-zero density), so the number to
watch is the validation MAE falling below ~2 and then grinding towards 1.

---

## 7. Things worth knowing before you trust the numbers

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
