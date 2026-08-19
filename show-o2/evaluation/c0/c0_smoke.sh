#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODE=${1:-dry}
SMOKE_IMAGE=${SMOKE_IMAGE:-"${SHOWO2_ROOT}/docs/mmu/pexels-muffin-1558665.jpg"}
SMOKE_VIDEO=${SMOKE_VIDEO:-"${SHOWO2_ROOT}/docs/videos/video1.mp4"}
SMOKE_TASK=${SMOKE_TASK:-mmstar}

case "${MODE}" in
    dry)
        "${C0_DIR}/c0_preflight.sh" runtime
        "${PYTHON}" "${C0_DIR}/smoke_adapter.py" --mode dry --image "${SMOKE_IMAGE}"
        ;;
    real)
        "${C0_DIR}/c0_preflight.sh" full
        "${PYTHON}" "${C0_DIR}/smoke_adapter.py" --mode real --image "${SMOKE_IMAGE}"
        ;;
    lmms)
        "${C0_DIR}/c0_smoke.sh" real
        smoke_output="${OUTPUT_ROOT}/smoke-$(timestamp_utc)"
        mkdir -p "${smoke_output}"
        "${LMMS_EVAL}" eval \
            --model showo2_qwen2_5 \
            --model_args "$(model_args)" \
            --tasks "${SMOKE_TASK}" \
            --batch_size 1 \
            --limit 1 \
            --log_samples \
            --output_path "${smoke_output}"
        ;;
    video-dry)
        "${C0_DIR}/c0_preflight.sh" runtime
        "${PYTHON}" "${C0_DIR}/smoke_video_adapter.py" --mode dry --video "${SMOKE_VIDEO}"
        ;;
    video-real)
        "${C0_DIR}/c0_preflight.sh" full
        require_file "${SHOWO2_VIDEO_CONFIG}"
        "${PYTHON}" "${C0_DIR}/smoke_video_adapter.py" --mode real --video "${SMOKE_VIDEO}"
        ;;
    *) die "Usage: $0 [dry|real|lmms|video-dry|video-real]" ;;
esac

