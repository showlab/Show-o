#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from lmms_eval.api.instance import Instance
from lmms_eval.models import get_model, get_model_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry", "real"), required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--question", default="Describe the main event in this video briefly.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    manifest = get_model_manifest("showo2_video")
    model_class = get_model("showo2_video", force_simple=True)
    kwargs = {
        "showo2_root": os.environ["SHOWO2_ROOT"],
        "config_file": os.environ["SHOWO2_VIDEO_CONFIG"],
        "pretrained": os.environ["SHOWO2_VIDEO_MODEL"],
        "vae_path": os.environ["SHOWO2_VAE_PATH"],
        "device": "cuda",
        "batch_size": 1,
        "weight_type": os.environ.get("WEIGHT_TYPE", "bfloat16"),
        "max_new_tokens": int(os.environ.get("SMOKE_MAX_NEW_TOKENS", "8")),
        "top_k": 1,
        "temperature": 0,
        "seed": int(os.environ.get("SEED", "0")),
        "num_video_frames": int(os.environ.get("SHOWO2_NUM_VIDEO_FRAMES", "32")),
        "video_token_size": int(os.environ.get("SHOWO2_VIDEO_TOKEN_SIZE", "14")),
        "dry_run": args.mode == "dry",
    }
    model = model_class(**kwargs)
    request = Instance(
        request_type="generate_until",
        arguments=(
            args.question,
            {"until": ["<STOP>"], "temperature": 0, "do_sample": False},
            lambda _doc: [str(video_path)],
            0,
            "showo2_video_smoke",
            "test",
        ),
        idx=0,
        metadata={"task": "showo2_video_smoke", "doc_id": 0, "repeats": 1},
        doc={"video": str(video_path)},
    )
    response = model.generate_until([request])
    if len(response) != 1 or not isinstance(response[0], str) or not response[0]:
        raise RuntimeError(f"Invalid video adapter response: {response!r}")
    if args.mode == "dry" and response[0] != "SHOWO2_DRY_RUN_OK":
        raise RuntimeError(f"Unexpected dry-run response: {response[0]!r}")
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": args.mode,
                "model_id": manifest.model_id,
                "adapter": f"{model_class.__module__}.{model_class.__name__}",
                "video": str(video_path),
                "frames": kwargs["num_video_frames"],
                "visual_tokens": kwargs["num_video_frames"] * kwargs["video_token_size"] ** 2,
                "response": response[0],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
