#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

MODE=${1:-runtime}
case "${MODE}" in
    structural|runtime|full) ;;
    *) die "Usage: $0 [structural|runtime|full]" ;;
esac

require_dir "${SHOWO2_ROOT}"
require_dir "${LMMS_EVAL_ROOT}"
require_file "${SHOWO2_CONFIG}"
require_file "${SHOWO2_VIDEO_CONFIG}"
require_file "${SHOWO2_ROOT}/models/modeling_showo2_qwen2_5.py"
require_file "${SHOWO2_ROOT}/lmms_eval_plugin/pyproject.toml"

if [[ "${MODE}" == "structural" ]]; then
    echo "C0 structural preflight: OK"
    exit 0
fi

require_executable "${PYTHON}"
require_executable "${LMMS_EVAL}"

"${PYTHON}" - <<'PY'
from importlib.metadata import version

expected = {
    "torch": "2.7.1",
    "torchvision": "0.22.1",
    "transformers": "4.47.0",
    "accelerate": "1.2.1",
    "fsspec": "2025.9.0",
    "lmms_eval": "0.7.2",
    "showo2-lmms-plugin": "0.2.0",
}
bad = []
for package, wanted in expected.items():
    found = version(package)
    print(f"{package}={found}")
    if found.split("+")[0] != wanted:
        bad.append(f"{package}: expected {wanted}, found {found}")
if bad:
    raise SystemExit("Version mismatch:\n  " + "\n  ".join(bad))
PY

"${PYTHON}" - <<'PY'
from lmms_eval.models import get_model, get_model_manifest

expected = {
    "showo2": ("showo2_qwen2_5", "ShowO2Qwen25"),
    "showo2_video": ("showo2_qwen2_5_video", "ShowO2Qwen25Video"),
}
for alias, (model_id, class_name) in expected.items():
    manifest = get_model_manifest(alias)
    model_class = get_model(alias, force_simple=True)
    if manifest.model_id != model_id or model_class.__name__ != class_name:
        raise SystemExit(
            f"Unexpected Show-o2 entry point for {alias}: "
            f"{manifest.model_id} / {model_class.__name__}"
        )
    print(
        "showo2_entry_point",
        manifest.model_id,
        f"{model_class.__module__}.{model_class.__name__}",
    )
PY

if [[ "${MODE}" == "runtime" ]]; then
    echo "C0 runtime preflight: OK"
    exit 0
fi

require_file "${SHOWO2_VAE_PATH}"
"${PYTHON}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; run the full preflight after attaching a GPU")
print("torch_cuda", torch.version.cuda)
print("compiled_arches", torch.cuda.get_arch_list())
print("cuda_devices", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    device = torch.device(f"cuda:{index}")
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    try:
        probe = torch.ones((16, 16), device=device, dtype=torch.bfloat16)
        torch.mm(probe, probe)
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        raise SystemExit(
            f"CUDA kernel probe failed on device {index} ({name}, sm_{capability[0]}{capability[1]}): {exc}"
        ) from exc
    print(index, name, "capability", capability, "bf16_matmul=OK")
PY

if [[ -e "${SHOWO2_MODEL}" ]]; then
    require_dir "${SHOWO2_MODEL}"
else
    echo "Show-o2 model is a Hugging Face id and will be resolved at load time: ${SHOWO2_MODEL}"
fi

echo "C0 full preflight: OK"

