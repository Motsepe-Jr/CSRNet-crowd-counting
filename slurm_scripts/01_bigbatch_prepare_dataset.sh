#!/bin/bash
#SBATCH --job-name=csrnet_data
#SBATCH --partition=bigbatch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-cluster}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-${PROJECT_DIR}/.conda-cluster}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_DIR}/ShanghaiTech_Crowd_Counting_Dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/data_splits}"
PART="${PART:-both}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SEED="${SEED:-42}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
TEST_LIMIT="${TEST_LIMIT:-}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"

export PYTHONUNBUFFERED=1

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

cd "${PROJECT_DIR}"
activate_env

ARGS=(
  --dataset-root "${DATASET_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --part "${PART}"
  --val-ratio "${VAL_RATIO}"
  --seed "${SEED}"
)

if [[ -n "${TRAIN_LIMIT}" ]]; then
  ARGS+=(--train-limit "${TRAIN_LIMIT}")
fi
if [[ -n "${TEST_LIMIT}" ]]; then
  ARGS+=(--test-limit "${TEST_LIMIT}")
fi
if [[ "${FORCE_REBUILD}" == "true" ]]; then
  ARGS+=(--force)
fi

python scripts/prepare_dataset.py "${ARGS[@]}"