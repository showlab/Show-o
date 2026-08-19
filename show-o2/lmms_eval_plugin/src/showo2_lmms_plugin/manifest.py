"""Entry-point manifest consumed by lmms-eval 0.7 and later."""

from lmms_eval.models.registry_v2 import ModelManifest

MODEL_MANIFESTS = (
    ModelManifest(
        model_id="showo2_qwen2_5",
        simple_class_path="showo2_lmms_plugin.model.ShowO2Qwen25",
        aliases=("showo2", "show-o2"),
    ),
    ModelManifest(
        model_id="showo2_qwen2_5_video",
        simple_class_path="showo2_lmms_plugin.model.ShowO2Qwen25Video",
        aliases=("showo2_video", "show-o2-video"),
    ),
)

