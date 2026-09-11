#!/bin/bash
#SBATCH --job-name=csrnet_eval
#SBATCH --partition=biggpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-cluster}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-${PROJECT_DIR}/.conda-cluster}"
DATA_SPLIT_DIR="${DATA_SPLIT_DIR:-${PROJECT_DIR}/data_splits}"
PART="${PART:-A}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE:-16}}"
WORKERS="${WORKERS:-4}"
GT_DOWNSAMPLE="${GT_DOWNSAMPLE:-}"
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

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/checkpoints/${DATA_PREFIX}}"
EVAL_JSON="${EVAL_JSON:-${DATA_SPLIT_DIR}/${DATA_PREFIX}_test.json}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/${TASK}model_best.pth.tar}"
SPLIT_NAME="${SPLIT_NAME:-${DATA_PREFIX}_test}"

export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"
activate_env

ARGS=(
  --json "${EVAL_JSON}"
  --checkpoint "${CHECKPOINT}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --split-name "${SPLIT_NAME}"
  --gt-downsample "${GT_DOWNSAMPLE}"
  --wandb-mode "${WANDB_MODE}"
)

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

python scripts/evaluate.py "${ARGS[@]}"