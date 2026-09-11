#!/bin/bash
#SBATCH --job-name=csrnet_train
#SBATCH --partition=biggpu
#SBATCH --exclude=mscluster110
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-cluster}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-${PROJECT_DIR}/.conda-cluster}"
DATA_SPLIT_DIR="${DATA_SPLIT_DIR:-${PROJECT_DIR}/data_splits}"
PART="${PART:-A}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WORKERS="${WORKERS:-4}"
LR="${LR:-1e-7}"
MOMENTUM="${MOMENTUM:-0.95}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
PRINT_FREQ="${PRINT_FREQ:-30}"
SEED="${SEED:-42}"
AMP="${AMP:-true}"
GT_DOWNSAMPLE="${GT_DOWNSAMPLE:-}"
PRECHECKPOINT="${PRECHECKPOINT:-}"
WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_TAGS="${WANDB_TAGS:-}"

activate_env() {
  if [[ -d "${CONDA_ENV_DIR}/conda-meta" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_DIR}"
    return
  fi
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    source "${VENV_DIR}/bin/activate"
    return
  fi
  if [[ -d "${VENV_DIR}/conda-meta" ]] && [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate "${VENV_DIR}"
    return
  fi
  echo "Could not activate environment at ${VENV_DIR} or ${CONDA_ENV_DIR}" >&2
  exit 1
}

case "${PART^^}" in
  A)
    DATA_PREFIX="part_A"
    TASK="${TASK:-partA_}"
    ;;
  B)
    DATA_PREFIX="part_B"
    TASK="${TASK:-partB_}"
    ;;
  UCSD)
    DATA_PREFIX="ucsd"
    TASK="${TASK:-ucsd_}"
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

TRAIN_JSON="${TRAIN_JSON:-${DATA_SPLIT_DIR}/${DATA_PREFIX}_train.json}"
VAL_JSON="${VAL_JSON:-${DATA_SPLIT_DIR}/${DATA_PREFIX}_val.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/checkpoints/${DATA_PREFIX}}"

export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"
activate_env
mkdir -p "${OUTPUT_DIR}"

ARGS=(
  --train-json "${TRAIN_JSON}"
  --val-json "${VAL_JSON}"
  --task "${OUTPUT_DIR}/${TASK}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --lr "${LR}"
  --momentum "${MOMENTUM}"
  --weight-decay "${WEIGHT_DECAY}"
  --print-freq "${PRINT_FREQ}"
  --seed "${SEED}"
  --gt-downsample "${GT_DOWNSAMPLE}"
  --wandb-mode "${WANDB_MODE}"
)

if [[ -n "${PRECHECKPOINT}" ]]; then
  ARGS+=(--pre "${PRECHECKPOINT}")
fi
if [[ "${AMP}" == "true" ]]; then
  ARGS+=(--amp)
fi
if [[ -n "${WANDB_PROJECT}" ]]; then
  ARGS+=(--wandb-project "${WANDB_PROJECT}")
fi
if [[ -n "${WANDB_GROUP}" ]]; then
  ARGS+=(--wandb-group "${WANDB_GROUP}")
fi
if [[ -n "${WANDB_RUN_NAME}" ]]; then
  ARGS+=(--wandb-run-name "${WANDB_RUN_NAME}")
fi
if [[ -n "${WANDB_TAGS}" ]]; then
  ARGS+=(--wandb-tags "${WANDB_TAGS}")
fi

python train.py "${ARGS[@]}"