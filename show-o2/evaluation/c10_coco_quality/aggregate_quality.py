#!/usr/bin/env python3
"""Validate and aggregate COCO quality metrics into one comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import utc_now, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmmd", type=Path, required=True)
    parser.add_argument("--fid-kid", type=Path, required=True)
    parser.add_argument("--image-reward", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_completed(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "completed":
        raise RuntimeError(f"metric is not complete: {path}")
    return value


def main() -> None:
    args = parse_args()
    cmmd = load_completed(args.cmmd)
    fid_kid = load_completed(args.fid_kid)
    reward = load_completed(args.image_reward)
    cmmd_names = set(cmmd["scores"])
    fid_names = set(fid_kid["scores"])
    reward_names = set(reward["scores"])
    if cmmd_names != fid_names or cmmd_names != reward_names:
        raise RuntimeError(
            f"metric variant mismatch: CMMD={cmmd_names}, FID={fid_names}, "
            f"ImageReward={reward_names}"
        )
    counts = {
        int(cmmd["images_per_set"]),
        int(fid_kid["images_per_set"]),
        int(reward["images_per_variant"]),
    }
    if len(counts) != 1:
        raise RuntimeError(f"metric image-count mismatch: {counts}")
    table = {
        name: {
            "cmmd_lower_is_better": float(cmmd["scores"][name]["cmmd"]),
            "fid_lower_is_better": float(fid_kid["scores"][name]["fid"]),
            "kid_x1000_lower_is_better": float(
                fid_kid["scores"][name]["kid_x1000"]
            ),
            "image_reward_higher_is_better": float(
                reward["scores"][name]["image_reward"]["mean"]
            ),
            "image_reward_bootstrap_95_ci": reward["scores"][name][
                "bootstrap_95_ci"
            ],
        }
        for name in sorted(cmmd_names)
    }
    output = {
        "status": "completed",
        "protocol": "showo2-coco5k-quality-aggregate-v1",
        "images_per_variant": counts.pop(),
        "metrics": table,
        "image_reward_pairwise": reward["pairwise"],
        "sources": {
            "cmmd": str(args.cmmd.resolve()),
            "fid_kid": str(args.fid_kid.resolve()),
            "image_reward": str(args.image_reward.resolve()),
        },
        "updated_at": utc_now(),
    }
    write_json(args.output, output)
    print(json.dumps(output, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
