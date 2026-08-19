#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_DIR=${1:-${RUN_DIR:-}}
if [[ -z "${RUN_DIR}" ]]; then
    RUN_DIR=$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'c0-*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)
fi
[[ -n "${RUN_DIR}" ]] || die "No C0 run directory was supplied or found under ${OUTPUT_ROOT}"
require_dir "${RUN_DIR}"
RUN_DIR=$(cd -- "${RUN_DIR}" && pwd)

MANIFEST="${RUN_DIR}/c0_manifest.txt"
{
    echo "created_utc=$(date -u --iso-8601=seconds)"
    echo "run_dir=${RUN_DIR}"
    echo "host=$(hostname)"
    echo "kernel=$(uname -srmo)"
    echo "showo2_root=${SHOWO2_ROOT}"
    echo "showo2_config=${SHOWO2_CONFIG}"
    echo "showo2_model=${SHOWO2_MODEL}"
    echo "showo2_vae_path=${SHOWO2_VAE_PATH}"
    echo "lmms_eval_root=${LMMS_EVAL_ROOT}"
    echo "geneval_root=${GENEVAL_ROOT}"
    echo "python=${PYTHON}"
    git -C "${WORKSPACE_ROOT}/Show-o" rev-parse HEAD 2>/dev/null | sed 's/^/showo_commit=/' || true
    git -C "${GENEVAL_ROOT}" rev-parse HEAD 2>/dev/null | sed 's/^/geneval_commit=/' || true
    if git -C "${LMMS_EVAL_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
        git -C "${LMMS_EVAL_ROOT}" rev-parse HEAD | sed 's/^/lmms_eval_commit=/'
    else
        echo "lmms_eval_source=main archive, version 0.7.2"
    fi
    if [[ -f "${SHOWO2_VAE_PATH}" ]]; then
        stat -c 'vae_size=%s%nvae_mtime=%y' "${SHOWO2_VAE_PATH}"
    fi
} >"${MANIFEST}"

"${PYTHON}" -m pip freeze >"${RUN_DIR}/pip-freeze.txt"
cp "${SHOWO2_CONFIG}" "${RUN_DIR}/showo2-config.yaml"

if [[ "${HASH_WEIGHTS:-0}" == "1" && -f "${SHOWO2_VAE_PATH}" ]]; then
    sha256sum "${SHOWO2_VAE_PATH}" >"${RUN_DIR}/weight-checksums.sha256"
fi

ARCHIVE_PATH=${ARCHIVE_PATH:-"${RUN_DIR}.tar.gz"}
tar -czf "${ARCHIVE_PATH}" -C "$(dirname -- "${RUN_DIR}")" "$(basename -- "${RUN_DIR}")"
echo "${ARCHIVE_PATH}"

