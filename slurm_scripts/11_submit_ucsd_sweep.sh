#!/bin/bash
# Submit a UCSD training sweep: one environment build, two dataset variants,
# then several training runs that differ only in the knobs under test. Every
# run reports to the same Weights & Biases group so the curves overlay.
#
#   WANDB_API_KEY=... ./slurm_scripts/11_submit_ucsd_sweep.sh
#
# Set SKIP_SETUP=true / SKIP_PREPARE=true to reuse an environment and data that
# are already in place.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"

VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-cluster}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-${PROJECT_DIR}/.conda-cluster}"
UCSD_ROOT="${UCSD_ROOT:-${PROJECT_DIR}/UCSD_Crowd_Counting_Dataset}"
DATA_SPLIT_DIR="${DATA_SPLIT_DIR:-${PROJECT_DIR}/data_splits}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_DIR}/checkpoints}"

EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
AMP="${AMP:-true}"
SKIP_SETUP="${SKIP_SETUP:-false}"
SKIP_PREPARE="${SKIP_PREPARE:-false}"

STAMP="$(date +%Y%m%d_%H%M%S)"
WANDB_PROJECT="${WANDB_PROJECT:-csrnet-crowd-counting}"
WANDB_GROUP="${WANDB_GROUP:-ucsd_sweep_${STAMP}}"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_MODE="${WANDB_MODE:-online}"
else
  WANDB_MODE="${WANDB_MODE:-disabled}"
  echo "note: WANDB_API_KEY is unset, so runs will not report to W&B" >&2
fi

export PROJECT_DIR VENV_DIR CONDA_ENV_DIR UCSD_ROOT DATA_SPLIT_DIR
export EPOCHS BATCH_SIZE WORKERS DEVICE AMP
export WANDB_API_KEY WANDB_PROJECT WANDB_GROUP WANDB_MODE
export PART=UCSD

# --------------------------------------------------------------- 00 setup
upstream=()
deps=""
if [[ "${SKIP_SETUP}" != "true" ]]; then
  setup_job=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/00_bigbatch_setup_venv.sh")
  echo "setup_job=${setup_job}"
  deps="--dependency=afterok:${setup_job}"
  upstream+=("${setup_job}")
fi

# ------------------------------------------------------ 01 dataset variants
# sigma 3 is the published CSRNet setting; sigma 8 widens each person's blob to
# roughly one output pixel, which spreads the gradient over ~8.5% of the target
# instead of ~2.2% and is the obvious suspect for a plateau.
if [[ "${SKIP_PREPARE}" != "true" ]]; then
  s3=$(UCSD_SIGMA=3.0 UCSD_PROCESSED_NAME=ucsd_processed UCSD_PREFIX=ucsd \
       sbatch --parsable ${deps} --export=ALL,UCSD_SIGMA=3.0,UCSD_PROCESSED_NAME=ucsd_processed,UCSD_PREFIX=ucsd \
       "${SCRIPT_DIR}/01_bigbatch_prepare_ucsd.sh")
  echo "prep_sigma3=${s3}"

  # The second prepare reuses the archives the first one downloaded.
  s8=$(sbatch --parsable --dependency=afterok:${s3} \
       --export=ALL,UCSD_SIGMA=8.0,UCSD_PROCESSED_NAME=ucsd_processed_s8,UCSD_PREFIX=ucsd_s8,DOWNLOAD=false \
       "${SCRIPT_DIR}/01_bigbatch_prepare_ucsd.sh")
  echo "prep_sigma8=${s8}"
  upstream+=("${s3}" "${s8}")
fi

# Training waits on everything upstream - the environment build as well as the
# data prep. Skipping one stage must not silently drop the other's dependency.
prep_deps=()
if (( ${#upstream[@]} > 0 )); then
  prep_deps=(--dependency=afterok:"$(IFS=:; echo "${upstream[*]}")")
fi

# ------------------------------------------------------------ 02 the sweep
# name | data prefix | optimizer | lr | extra env
RUNS=(
  "sgd_sigma3|ucsd|sgd|1e-6|LR_SCHEDULE=cosine CLIP_GRAD=0"
  "adam_sigma3|ucsd|adam|1e-5|LR_SCHEDULE=cosine CLIP_GRAD=5"
  "adam_sigma8|ucsd_s8|adam|1e-5|LR_SCHEDULE=cosine CLIP_GRAD=5"
  "adam_sigma8_aug|ucsd_s8|adam|1e-5|LR_SCHEDULE=cosine CLIP_GRAD=5 AUG_BRIGHTNESS=0.2 AUG_CONTRAST=0.2 AUG_NOISE=3 MIN_CROP_DENSITY=2"
)

for spec in "${RUNS[@]}"; do
  IFS='|' read -r name prefix opt lr extra <<< "${spec}"
  out="${CHECKPOINT_ROOT}/ucsd_${STAMP}_${name}"
  # WANDB_TAGS holds commas, which would split SLURM's comma-separated
  # --export list, so pass both through the inherited environment.
  export WANDB_TAGS="ucsd,sweep,${name}"
  export WANDB_RUN_NAME="${name}"
  job=$(sbatch --parsable "${prep_deps[@]}" \
    --job-name="csr_${name}" \
    --export=ALL,DATA_PREFIX_OVERRIDE="${prefix}",OPTIMIZER="${opt}",LR="${lr}",OUTPUT_DIR="${out}",TASK="${name}_",$(echo "${extra}" | tr ' ' ',') \
    "${SCRIPT_DIR}/02_biggpu_train.sh")
  echo "train_${name}=${job}   -> ${out}"
done

echo
echo "wandb_group=${WANDB_GROUP}"
echo "project=${WANDB_PROJECT}"
echo "watch with: squeue -u \$USER"
