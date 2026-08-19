#!/usr/bin/env bash

C0_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export SHOWO2_ROOT=${SHOWO2_ROOT:-$(cd -- "${C0_DIR}/../.." && pwd)}
export WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(cd -- "${SHOWO2_ROOT}/../.." && pwd)}
export LMMS_EVAL_ROOT=${LMMS_EVAL_ROOT:-"${WORKSPACE_ROOT}/lmms-eval"}
export GENEVAL_ROOT=${GENEVAL_ROOT:-"${WORKSPACE_ROOT}/geneval"}
export SHOWO2_EVAL_ENV=${SHOWO2_EVAL_ENV:-"${WORKSPACE_ROOT}/venvs/showo2-eval"}
export DPG_SCORE_ENV=${DPG_SCORE_ENV:-"${WORKSPACE_ROOT}/venvs/dpg-score"}
export GENEVAL_SCORE_ENV=${GENEVAL_SCORE_ENV:-"${WORKSPACE_ROOT}/venvs/geneval-score"}
export PYTHON=${PYTHON:-"${SHOWO2_EVAL_ENV}/bin/python"}
export LMMS_EVAL=${LMMS_EVAL:-"${SHOWO2_EVAL_ENV}/bin/lmms-eval"}
export ACCELERATE=${ACCELERATE:-"${SHOWO2_EVAL_ENV}/bin/accelerate"}
export DPG_PYTHON=${DPG_PYTHON:-"${DPG_SCORE_ENV}/bin/python"}
export DPG_ACCELERATE=${DPG_ACCELERATE:-"${DPG_SCORE_ENV}/bin/accelerate"}
export DPG_MPLUG_PATH=${DPG_MPLUG_PATH:-"${WORKSPACE_ROOT}/models/dpg/mplug_vqa"}
export GENEVAL_PYTHON=${GENEVAL_PYTHON:-"${GENEVAL_SCORE_ENV}/bin/python"}
export GENEVAL_MODEL_PATH=${GENEVAL_MODEL_PATH:-"${WORKSPACE_ROOT}/models/geneval"}
export GENEVAL_MODEL_CONFIG=${GENEVAL_MODEL_CONFIG:-"${WORKSPACE_ROOT}/mmdetection-2.28.2/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"}
export GENEVAL_CLIP_PATH=${GENEVAL_CLIP_PATH:-"${WORKSPACE_ROOT}/models/geneval/clip/ViT-L-14.state_dict.pt"}
export SHOWO2_CONFIG=${SHOWO2_CONFIG:-"${SHOWO2_ROOT}/configs/showo2_1.5b_demo_432x432.yaml"}
export SHOWO2_MODEL=${SHOWO2_MODEL:-"showlab/show-o2-1.5B"}
export SHOWO2_VAE_PATH=${SHOWO2_VAE_PATH:-"${SHOWO2_ROOT}/Wan2.1_VAE.pth"}
export OUTPUT_ROOT=${OUTPUT_ROOT:-"${WORKSPACE_ROOT}/runs/showo2_c0"}
export HF_HOME=${HF_HOME:-"${WORKSPACE_ROOT}/cache/huggingface"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"${HF_HOME}/datasets"}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_MODE=${WANDB_MODE:-disabled}
export SHOWO2_VIDEO_CONFIG=${SHOWO2_VIDEO_CONFIG:-"${SHOWO2_ROOT}/configs/showo2_1.5b_demo_video_understanding.yaml"}
export SHOWO2_VIDEO_MODEL=${SHOWO2_VIDEO_MODEL:-"showlab/show-o2-1.5B-w-video-und"}
export SHOWO2_NUM_VIDEO_FRAMES=${SHOWO2_NUM_VIDEO_FRAMES:-32}
export SHOWO2_VIDEO_TOKEN_SIZE=${SHOWO2_VIDEO_TOKEN_SIZE:-14}
export SHOWO2_TASKS_PATH=${SHOWO2_TASKS_PATH:-"${C0_DIR}/tasks"}
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS=1
fi
if [[ ! "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export MKL_NUM_THREADS=1
fi

die() {
    echo "ERROR: $*" >&2
    exit 2
}

timestamp_utc() {
    date -u +%Y%m%dT%H%M%SZ
}

require_file() {
    [[ -f "$1" ]] || die "Required file not found: $1"
}

require_dir() {
    [[ -d "$1" ]] || die "Required directory not found: $1"
}

require_executable() {
    [[ -x "$1" ]] || die "Required executable not found: $1"
}

model_args() {
    local args
    args="showo2_root=${SHOWO2_ROOT},config_file=${SHOWO2_CONFIG},pretrained=${SHOWO2_MODEL},vae_path=${SHOWO2_VAE_PATH},device=cuda,weight_type=${WEIGHT_TYPE:-bfloat16},max_new_tokens=${MAX_NEW_TOKENS:-300},top_k=${TOP_K:-1},temperature=${TEMPERATURE:-0},seed=${SEED:-0}"
    printf '%s' "${args}"
}

video_model_args() {
    local args
    args="showo2_root=${SHOWO2_ROOT},config_file=${SHOWO2_VIDEO_CONFIG},pretrained=${SHOWO2_VIDEO_MODEL},vae_path=${SHOWO2_VAE_PATH},device=cuda,weight_type=${WEIGHT_TYPE:-bfloat16},max_new_tokens=${MAX_NEW_TOKENS:-300},top_k=${TOP_K:-1},temperature=${TEMPERATURE:-0},seed=${SEED:-0},num_video_frames=${SHOWO2_NUM_VIDEO_FRAMES},video_token_size=${SHOWO2_VIDEO_TOKEN_SIZE}"
    printf '%s' "${args}"
}

visible_gpu_ids() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES}"
        return
    fi
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null
}

