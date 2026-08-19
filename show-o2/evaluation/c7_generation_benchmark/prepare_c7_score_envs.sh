#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${WORKSPACE:-/root/autodl-tmp}
BASE_PYTHON=${BASE_PYTHON:-/root/miniconda3/bin/python}
VENV_ROOT=${VENV_ROOT:-${WORKSPACE}/venvs}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-${WORKSPACE}/tmp/pip-cache-c7}
export PIP_DISABLE_PIP_VERSION_CHECK=1

mkdir -p "${VENV_ROOT}" "${PIP_CACHE_DIR}"

create_venv() {
    local path=$1
    if [[ ! -x "${path}/bin/python" ]]; then
        "${BASE_PYTHON}" -m venv --system-site-packages "${path}"
    fi
}

VQA_ENV=${VENV_ROOT}/showo2-score-vqa
REWARD_ENV=${VENV_ROOT}/showo2-score-reward
FID_ENV=${VENV_ROOT}/showo2-score-fid

create_venv "${VQA_ENV}"
"${VQA_ENV}/bin/python" -m pip install --no-deps 't2v-metrics==3.0'
"${VQA_ENV}/bin/python" -m pip install \
    'transformers==4.49.0' \
    'accelerate>=1.2,<2' \
    'qwen-vl-utils>=0.0.8' \
    'decord==0.6.0' \
    'av>=13,<16' \
    'ftfy>=6.1.1'

create_venv "${REWARD_ENV}"
"${REWARD_ENV}/bin/python" -m pip install --no-deps 'image-reward==1.5'
"${REWARD_ENV}/bin/python" -m pip install \
    'transformers==4.47.0' \
    'timm==0.6.13' \
    'fairscale==0.4.13' \
    'openai-clip==1.0.1' \
    'accelerate>=1.2,<2' \
    'datasets==3.2.0' \
    'diffusers==0.32.2' \
    'huggingface-hub>=0.24,<1' \
    'safetensors>=0.4,<1'

create_venv "${FID_ENV}"
"${FID_ENV}/bin/python" -m pip install 'clean-fid==0.1.35' 'scipy>=1.11,<2'

"${VQA_ENV}/bin/python" -c 'import torch, transformers; print("vqa", torch.__version__, transformers.__version__)'
"${REWARD_ENV}/bin/python" -c 'import torch, transformers; print("reward", torch.__version__, transformers.__version__)'
"${FID_ENV}/bin/python" -c 'import torch, cleanfid, scipy; print("fid", torch.__version__, scipy.__version__)'

