#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image

from lmms_eval.api.instance import Instance
from lmms_eval.models import get_model, get_model_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry", "real"), required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default="Describe this image briefly.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    manifest = get_model_manifest("showo2")
    model_class = get_model("showo2", force_simple=True)
    kwargs = {
        "showo2_root": os.environ["SHOWO2_ROOT"],
        "config_file": os.environ["SHOWO2_CONFIG"],
        "pretrained": os.environ["SHOWO2_MODEL"],
        "vae_path": os.environ["SHOWO2_VAE_PATH"],
        "device": "cuda",
        "batch_size": 1,
        "weight_type": os.environ.get("WEIGHT_TYPE", "bfloat16"),
        "max_new_tokens": int(os.environ.get("SMOKE_MAX_NEW_TOKENS", "8")),
        "top_k": 1,
        "seed": int(os.environ.get("SEED", "0")),
        "dry_run": args.mode == "dry",
    }
    model = model_class(**kwargs)
    request = Instance(
        request_type="generate_until",
        arguments=(args.question, {"until": ["<STOP>"]}, lambda _doc: [image], 0, "showo2_smoke", "test"),
        idx=0,
        metadata={"task": "showo2_smoke", "doc_id": 0, "repeats": 1},
        doc={"image": str(image_path)},
    )
    response = model.generate_until([request])
    if len(response) != 1 or not isinstance(response[0], str) or not response[0]:
        raise RuntimeError(f"Invalid adapter response: {response!r}")
    if args.mode == "dry" and response[0] != "SHOWO2_DRY_RUN_OK":
        raise RuntimeError(f"Unexpected dry-run response: {response[0]!r}")
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": args.mode,
                "model_id": manifest.model_id,
                "adapter": f"{model_class.__module__}.{model_class.__name__}",
                "image": str(image_path),
                "response": response[0],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

