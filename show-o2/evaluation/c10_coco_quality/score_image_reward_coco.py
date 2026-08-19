#!/usr/bin/env python3
"""Score paired COCO generations with the cached ImageReward-v1.0 model."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import ImageReward as RM
import numpy as np
import torch
from PIL import Image
from transformers import BertTokenizer

from common import read_coco_manifest, utc_now, validate_flat_image_dir, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--med-config", type=Path, required=True)
    parser.add_argument("--bert-tokenizer", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_variants(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"variant must be NAME=PATH: {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        path = Path(raw_path).resolve()
        if not name or name in result:
            raise ValueError(f"invalid or duplicate variant: {name!r}")
        result[name] = path
    return result


def patch_local_tokenizer(path: Path) -> None:
    from ImageReward.models.BLIP import blip_pretrain

    def init_tokenizer() -> BertTokenizer:
        tokenizer = BertTokenizer.from_pretrained(str(path), local_files_only=True)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
        return tokenizer

    blip_pretrain.init_tokenizer = init_tokenizer


def score_batch(model: Any, records: list[dict[str, Any]], device: torch.device) -> list[float]:
    images = []
    for record in records:
        with Image.open(record["image_path"]) as image:
            images.append(model.preprocess(image.convert("RGB")))
    image_tensor = torch.stack(images).to(device)
    text_input = model.blip.tokenizer(
        [record["prompt"] for record in records],
        padding="max_length",
        truncation=True,
        max_length=35,
        return_tensors="pt",
    ).to(device)
    image_embeds = model.blip.visual_encoder(image_tensor)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
    text_output = model.blip.text_encoder(
        text_input.input_ids,
        attention_mask=text_input.attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )
    features = text_output.last_hidden_state[:, 0, :].float()
    rewards = (model.mlp(features) - model.mean) / model.std
    return [float(value) for value in rewards.squeeze(-1).cpu()]


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "std": float(statistics.pstdev(values)),
    }


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draw_means = []
    chunk = 128
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        draw_means.extend(np.mean(values[indices], axis=1).tolist())
    return [float(value) for value in np.quantile(draw_means, [0.025, 0.975])]


def main() -> None:
    args = parse_args()
    rows = read_coco_manifest(args.manifest)
    variants = parse_variants(args.variant)
    filenames = [f"{index:05d}.png" for index in range(len(rows))]
    for root in variants.values():
        validate_flat_image_dir(root, filenames)
    for path in (args.checkpoint, args.med_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.bert_tokenizer.is_dir():
        raise FileNotFoundError(args.bert_tokenizer)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ImageReward scoring requires CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    total = len(rows) * len(variants)
    write_json(
        status_path,
        {
            "status": "running",
            "phase": "loading_model",
            "total_images": total,
            "completed_images": 0,
            "percent": 0.0,
            "updated_at": utc_now(),
        },
    )
    started = time.perf_counter()
    try:
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        patch_local_tokenizer(args.bert_tokenizer)
        model = RM.load(
            str(args.checkpoint), device=device, med_config=str(args.med_config)
        )
        model.eval().requires_grad_(False)
        ordered = [
            {
                "variant": name,
                "index": index,
                "prompt": str(row["prompt"]),
                "seed": int(row["seed"]),
                "coco_image_id": str(row["coco_image_id"]),
                "image_path": str((root / f"{index:05d}.png").resolve()),
            }
            for index, row in enumerate(rows)
            for name, root in variants.items()
        ]
        scored = []
        with torch.inference_mode():
            for start in range(0, len(ordered), args.batch_size):
                batch = ordered[start : start + args.batch_size]
                rewards = score_batch(model, batch, device)
                scored.extend(
                    {**record, "image_reward": reward}
                    for record, reward in zip(batch, rewards, strict=True)
                )
                write_json(
                    status_path,
                    {
                        "status": "running",
                        "phase": "scoring",
                        "total_images": total,
                        "completed_images": len(scored),
                        "percent": 100.0 * len(scored) / total,
                        "updated_at": utc_now(),
                    },
                )
        write_jsonl(args.output_dir / "per_image_scores.jsonl", scored)
        by_variant = {
            name: np.asarray(
                [
                    record["image_reward"]
                    for record in scored
                    if record["variant"] == name
                ],
                dtype=np.float64,
            )
            for name in variants
        }
        scores = {
            name: {
                "images": len(values),
                "image_reward": aggregate(values.tolist()),
                "bootstrap_95_ci": bootstrap_ci(
                    values, args.bootstrap_samples, args.bootstrap_seed + offset
                ),
            }
            for offset, (name, values) in enumerate(by_variant.items())
        }
        pairwise = {}
        for offset, (left, right) in enumerate(itertools.combinations(variants, 2)):
            delta = by_variant[right] - by_variant[left]
            pairwise[f"{left}__vs__{right}"] = {
                "delta_right_minus_left": float(np.mean(delta)),
                "bootstrap_95_ci": bootstrap_ci(
                    delta, args.bootstrap_samples, args.bootstrap_seed + 100 + offset
                ),
                "right_wins": int(np.sum(delta > 1e-12)),
                "left_wins": int(np.sum(delta < -1e-12)),
                "ties": int(np.sum(np.abs(delta) <= 1e-12)),
            }
        summary = {
            "status": "completed",
            "protocol": "showo2-coco5k-imagereward-paired-v1",
            "images_per_variant": len(rows),
            "model": "ImageReward-v1.0",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "scores": scores,
            "pairwise": pairwise,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0),
            },
            "runtime_seconds": time.perf_counter() - started,
            "updated_at": utc_now(),
        }
        write_json(args.output_dir / "summary.json", summary)
        write_json(
            status_path,
            {
                "status": "completed",
                "phase": "completed",
                "total_images": total,
                "completed_images": total,
                "percent": 100.0,
                "updated_at": utc_now(),
            },
        )
        (args.output_dir / "SCORING_COMPLETED").touch()
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    except BaseException as error:
        write_json(
            status_path,
            {
                "status": "failed",
                "phase": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "updated_at": utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    main()
