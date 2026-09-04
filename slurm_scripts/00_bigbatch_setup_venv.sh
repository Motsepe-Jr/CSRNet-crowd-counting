#!/bin/bash
#SBATCH --job-name=csrnet_setup
#SBATCH --partition=bigbatch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-cluster}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-${PROJECT_DIR}/.conda-cluster}"
PYTHON_BIN="${PYTHON_BIN:-}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch/csrnet}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
PREFER_CONDA="${PREFER_CONDA:-true}"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1
export TORCH_HOME

cd "${PROJECT_DIR}"

use_conda=false

if [[ "${PREFER_CONDA}" == "true" ]] && [[ -f "${CONDA_SH}" ]]; then
	use_conda=true
fi

is_supported_python() {
	local candidate="$1"
	"${candidate}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)
PY
}

if [[ -n "${PYTHON_BIN}" ]] && ! is_supported_python "${PYTHON_BIN}"; then
	echo "Preset PYTHON_BIN=${PYTHON_BIN} is not supported for torch cu121 wheels; falling back to auto-detect/conda."
	PYTHON_BIN=""
fi

if [[ "${use_conda}" == "false" ]] && [[ -z "${PYTHON_BIN}" ]]; then
	for candidate in python3.12 python3.11 python3.10 python3; do
		if command -v "${candidate}" >/dev/null 2>&1; then
			if is_supported_python "${candidate}"; then
				PYTHON_BIN="${candidate}"
				break
			fi
		fi
	done
fi

if [[ "${use_conda}" == "false" ]] && [[ -z "${PYTHON_BIN}" ]]; then
	if [[ -f "${CONDA_SH}" ]]; then
		use_conda=true
	else
		echo "Could not find a supported Python interpreter and conda is unavailable. Need Python 3.10, 3.11, or 3.12 for torch cu121 wheels." >&2
		exit 1
	fi
fi

if [[ "${use_conda}" == "true" ]]; then
	echo "Using conda fallback environment at ${CONDA_ENV_DIR}"
	mkdir -p "$(dirname -- "${CONDA_ENV_DIR}")" "${TORCH_HOME}"
	source "${CONDA_SH}"
	if [[ ! -x "${CONDA_ENV_DIR}/bin/python" ]]; then
		conda create -y -p "${CONDA_ENV_DIR}" python=3.11
	fi
	conda activate "${CONDA_ENV_DIR}"
else
	echo "Using Python interpreter: ${PYTHON_BIN}"
	mkdir -p "$(dirname -- "${VENV_DIR}")" "${TORCH_HOME}"
	"${PYTHON_BIN}" --version
	"${PYTHON_BIN}" -m venv "${VENV_DIR}"
	source "${VENV_DIR}/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url "${PYTORCH_INDEX_URL}" "torch>=2.2,<2.3" "torchvision>=0.17,<0.18"
python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless >/dev/null 2>&1 || true
python -m pip install -r requirements.txt
python -m pip install --upgrade --force-reinstall "numpy>=1.26,<2" "scipy>=1.11" "opencv-python-headless>=4.9"
python -m pip install "wandb>=0.17,<1"

python - <<'PY'
import torch
import torchvision
import numpy
import cv2

print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("numpy", numpy.__version__)
print("cv2", cv2.__version__)
PY

python - <<'PY'
from torchvision.models import VGG16_Weights, vgg16

model = vgg16(weights=VGG16_Weights.DEFAULT)
print("cached_vgg16", tuple(model.features[0].weight.shape))
PY