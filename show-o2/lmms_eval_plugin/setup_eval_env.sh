#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SHOWO2_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(cd -- "${SHOWO2_ROOT}/../.." && pwd)}
LMMS_EVAL_ROOT=${LMMS_EVAL_ROOT:-"${WORKSPACE_ROOT}/lmms-eval"}
ENV_DIR=${SHOWO2_EVAL_ENV:-"${WORKSPACE_ROOT}/venvs/showo2-eval"}
PYTHON_BOOTSTRAP=${PYTHON_BOOTSTRAP:-"${WORKSPACE_ROOT}/venvs/bfcl/bin/python"}
CONSTRAINTS=${CONSTRAINTS:-"${SCRIPT_DIR}/constraints-eval.txt"}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-"https://download.pytorch.org/whl/cu128"}
TORCH_SPEC=${TORCH_SPEC:-"torch==2.7.1+cu128"}
TORCHVISION_SPEC=${TORCHVISION_SPEC:-"torchvision==0.22.1+cu128"}

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS=1
fi
if [[ ! "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export MKL_NUM_THREADS=1
fi

if [[ ! -f "${LMMS_EVAL_ROOT}/pyproject.toml" ]]; then
    echo "ERROR: lmms-eval source not found: ${LMMS_EVAL_ROOT}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BOOTSTRAP}" ]]; then
    echo "ERROR: Python bootstrap not found: ${PYTHON_BOOTSTRAP}" >&2
    exit 2
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    echo "Creating isolated Python environment: ${ENV_DIR}"
    "${PYTHON_BOOTSTRAP}" -m venv "${ENV_DIR}"
fi

PYTHON="${ENV_DIR}/bin/python"
"${PYTHON}" -m pip install --upgrade "pip<26" setuptools wheel
"${PYTHON}" -m pip install --no-cache-dir --upgrade \
    --index-url "${TORCH_INDEX_URL}" \
    "${TORCH_SPEC}" \
    "${TORCHVISION_SPEC}"
"${PYTHON}" -m pip install -c "${CONSTRAINTS}" -e "${LMMS_EVAL_ROOT}"
"${PYTHON}" -m pip install --no-cache-dir -c "${CONSTRAINTS}" \
    diffusers==0.31.0 \
    omegaconf \
    torchdiffeq \
    timm==1.0.12 \
    einops==0.8.0
"${PYTHON}" -m pip install --no-deps -e "${SCRIPT_DIR}"
echo "Environment ready: ${ENV_DIR}"
"${PYTHON}" - <<'PY'
import accelerate
import diffusers
import lmms_eval
import torch
import transformers

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("diffusers", diffusers.__version__)
print("accelerate", accelerate.__version__)
print("lmms_eval", getattr(lmms_eval, "__version__", "source checkout"))
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    probe = torch.ones((16, 16), device=device, dtype=torch.bfloat16)
    torch.mm(probe, probe)
    torch.cuda.synchronize(device)
    print(
        "cuda_probe",
        torch.cuda.get_device_name(device),
        "capability",
        torch.cuda.get_device_capability(device),
        "OK",
    )
else:
    print("cuda_probe skipped: no visible CUDA device")
PY
