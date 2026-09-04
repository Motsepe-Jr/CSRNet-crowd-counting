#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-cluster}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-${PROJECT_DIR}/.conda-cluster}"
PYTHON_BIN="${PYTHON_BIN:-}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

PART="${PART:-A}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SEED="${SEED:-42}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
TEST_LIMIT="${TEST_LIMIT:-}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
WORKERS="${WORKERS:-4}"
LR="${LR:-1e-7}"
MOMENTUM="${MOMENTUM:-0.95}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
PRINT_FREQ="${PRINT_FREQ:-30}"
AMP="${AMP:-true}"
PRECHECKPOINT="${PRECHECKPOINT:-}"
GT_DOWNSAMPLE="${GT_DOWNSAMPLE:-}"

# UCSD-only knobs (see scripts/prepare_ucsd.py)
UCSD_SCALE="${UCSD_SCALE:-4}"
UCSD_SIGMA="${UCSD_SIGMA:-3.0}"
UCSD_ROI="${UCSD_ROI:-mask}"
UCSD_SPLIT="${UCSD_SPLIT:-standard}"
UCSD_IMAGE_FORMAT="${UCSD_IMAGE_FORMAT:-png}"
VAL_MODE="${VAL_MODE:-contiguous}"
DOWNLOAD="${DOWNLOAD:-true}"

PREP_SCRIPT="01_bigbatch_prepare_dataset.sh"

case "${PART^^}" in
  A)
    DATA_PREFIX="part_A"
    TASK="partA_"
    ;;
  B)
    DATA_PREFIX="part_B"
    TASK="partB_"
    ;;
  UCSD)
    DATA_PREFIX="ucsd"
    TASK="ucsd_"
    PREP_SCRIPT="01_bigbatch_prepare_ucsd.sh"
    # UCSD density maps use a tight sigma=3 kernel, so area-averaging the
    # ground truth down to the 1/8 output stride keeps the count exact.
    GT_DOWNSAMPLE="${GT_DOWNSAMPLE:-area}"
    ;;
  *)
    echo "Unsupported PART=${PART}" >&2
    exit 1
    ;;
esac

GT_DOWNSAMPLE="${GT_DOWNSAMPLE:-cubic}"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_DIR}/ShanghaiTech_Crowd_Counting_Dataset}"
UCSD_ROOT="${UCSD_ROOT:-${PROJECT_DIR}/UCSD_Crowd_Counting_Dataset}"
DATA_SPLIT_DIR="${DATA_SPLIT_DIR:-${PROJECT_DIR}/data_splits}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_DIR}/checkpoints}"
RUN_STEM="${RUN_STEM:-csrnet_${DATA_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/${RUN_STEM}}"
TRAIN_JSON="${TRAIN_JSON:-${DATA_SPLIT_DIR}/${DATA_PREFIX}_train.json}"
VAL_JSON="${VAL_JSON:-${DATA_SPLIT_DIR}/${DATA_PREFIX}_val.json}"
EVAL_JSON="${EVAL_JSON:-${DATA_SPLIT_DIR}/${DATA_PREFIX}_test.json}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/${TASK}model_best.pth.tar}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_MODE="${WANDB_MODE:-online}"
else
  WANDB_MODE="${WANDB_MODE:-disabled}"
fi
WANDB_PROJECT="${WANDB_PROJECT:-csr-net-cluster}"
WANDB_GROUP="${WANDB_GROUP:-csrnet_${DATA_PREFIX}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_STEM}}"
WANDB_TAGS="${WANDB_TAGS:-cluster,csrnet,${DATA_PREFIX}}"

export PROJECT_DIR VENV_DIR CONDA_ENV_DIR PYTHON_BIN PYTORCH_INDEX_URL
export PART DATASET_ROOT DATA_SPLIT_DIR VAL_RATIO SEED TRAIN_LIMIT TEST_LIMIT FORCE_REBUILD
export DEVICE EPOCHS BATCH_SIZE EVAL_BATCH_SIZE WORKERS LR MOMENTUM WEIGHT_DECAY PRINT_FREQ AMP PRECHECKPOINT
export GT_DOWNSAMPLE UCSD_ROOT UCSD_SCALE UCSD_SIGMA UCSD_ROI UCSD_SPLIT UCSD_IMAGE_FORMAT VAL_MODE DOWNLOAD
export TRAIN_JSON VAL_JSON EVAL_JSON OUTPUT_DIR CHECKPOINT TASK
export WANDB_MODE WANDB_PROJECT WANDB_GROUP WANDB_RUN_NAME WANDB_TAGS

setup_job=$(sbatch --parsable --export=ALL "${SCRIPT_DIR}/00_bigbatch_setup_venv.sh")
prep_job=$(sbatch --parsable --dependency=afterok:${setup_job} --export=ALL "${SCRIPT_DIR}/${PREP_SCRIPT}")
train_job=$(sbatch --parsable --dependency=afterok:${prep_job} --export=ALL "${SCRIPT_DIR}/02_biggpu_train.sh")
eval_job=$(sbatch --parsable --dependency=afterok:${train_job} --export=ALL "${SCRIPT_DIR}/03_biggpu_eval.sh")

echo "setup_job=${setup_job}"
echo "prep_job=${prep_job}"
echo "train_job=${train_job}"
echo "eval_job=${eval_job}"
echo "run_stem=${RUN_STEM}"
echo "checkpoint=${CHECKPOINT}"