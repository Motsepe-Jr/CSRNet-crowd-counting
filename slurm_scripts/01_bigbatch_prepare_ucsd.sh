#!/bin/bash
#SBATCH --job-name=csrnet_ucsd_data
#SBATCH --partition=bigbatch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-cluster}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-${PROJECT_DIR}/.conda-cluster}"
UCSD_ROOT="${UCSD_ROOT:-${PROJECT_DIR}/UCSD_Crowd_Counting_Dataset}"
OUTPUT_DIR="${DATA_SPLIT_DIR:-${PROJECT_DIR}/data_splits}"
UCSD_SCALE="${UCSD_SCALE:-4}"
UCSD_SIGMA="${UCSD_SIGMA:-3.0}"
UCSD_ROI="${UCSD_ROI:-mask}"
UCSD_SPLIT="${UCSD_SPLIT:-standard}"
UCSD_IMAGE_FORMAT="${UCSD_IMAGE_FORMAT:-png}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_MODE="${VAL_MODE:-contiguous}"
SEED="${SEED:-42}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"
DOWNLOAD="${DOWNLOAD:-true}"

UCSD_BASE_URL="http://www.svcl.ucsd.edu/projects/peoplecnt/db"

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

# The UCSD host is plain HTTP only (no TLS) - curl -C - resumes a partial file.
if [[ "${DOWNLOAD}" == "true" ]]; then
  mkdir -p "${UCSD_ROOT}/raw"
  for archive in ucsdpeds.zip vidf-cvpr.zip; do
    target="${UCSD_ROOT}/raw/${archive}"
    if [[ ! -s "${target}" ]] || [[ "${FORCE_REBUILD}" == "true" ]]; then
      echo "downloading ${archive}"
      curl -fSL -C - --retry 5 --retry-delay 5 -o "${target}" "${UCSD_BASE_URL}/${archive}"
    else
      echo "have ${archive} already"
    fi
  done
fi

ARGS=(
  --dataset-root "${UCSD_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --scale "${UCSD_SCALE}"
  --sigma "${UCSD_SIGMA}"
  --roi "${UCSD_ROI}"
  --split "${UCSD_SPLIT}"
  --val-ratio "${VAL_RATIO}"
  --val-mode "${VAL_MODE}"
  --seed "${SEED}"
  --image-format "${UCSD_IMAGE_FORMAT}"
)

if [[ "${FORCE_REBUILD}" == "true" ]]; then
  ARGS+=(--force)
fi

python scripts/prepare_ucsd.py "${ARGS[@]}"
