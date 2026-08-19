#!/usr/bin/env python3
"""Score generated images with OpenCLIP and relation counterfactual prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image


RELATION_SWAP = {
    "left of": "right of",
    "right of": "left of",
    "above": "below",
    "below": "above",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--clip-checkpoint",
        type=Path,
        default=Path(
            "/root/autodl-tmp/models/geneval/clip/ViT-L-14.state_dict.pt"
        ),
    )
    parser.add_argument("--clip-arch", default="ViT-L-14")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    parser.add_argument("--limit-images", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relation_from_metadata(metadata: dict[str, Any]) -> str:
    relations = [
        str(item["position"][0])
        for item in metadata.get("include", [])
        if "position" in item
    ]
    if len(relations) != 1 or relations[0] not in RELATION_SWAP:
        raise ValueError(f"expected one supported position relation, got {relations}")
    return relations[0]


def counterfactual_prompt(prompt: str, relation: str) -> str:
    if prompt.count(relation) != 1:
        raise ValueError(
            f"prompt must contain relation exactly once: relation={relation!r} prompt={prompt!r}"
        )
    return prompt.replace(relation, RELATION_SWAP[relation], 1)


def collect_images(root: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for prompt_dir in sorted(root.iterdir()):
        if not prompt_dir.is_dir() or not prompt_dir.name.isdigit():
            continue
        metadata_path = prompt_dir / "metadata.jsonl"
        samples_dir = prompt_dir / "samples"
        if not metadata_path.is_file() or not samples_dir.is_dir():
            raise FileNotFoundError(f"incomplete GenEval prompt directory: {prompt_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prompt = str(metadata["prompt"])
        relation = relation_from_metadata(metadata)
        swapped = counterfactual_prompt(prompt, relation)
        for image_path in sorted(samples_dir.glob("*.png")):
            if not image_path.stem.isdigit():
                continue
            records.append(
                {
                    "prompt_id": int(prompt_dir.name),
                    "sample_index": int(image_path.stem),
                    "image_path": str(image_path.resolve()),
                    "prompt": prompt,
                    "counterfactual_prompt": swapped,
                    "relation": relation,
                }
            )
            if limit is not None and len(records) >= limit:
                return records
    return records


def cluster_bootstrap_ci(
    records: list[dict[str, Any]],
    field: str,
    samples: int,
    seed: int,
) -> list[float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        grouped[int(record["prompt_id"])].append(float(record[field]))
    cluster_means = np.asarray(
        [np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = rng.choice(cluster_means, size=(samples, len(cluster_means)), replace=True)
    means = draws.mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def aggregate(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [float(record[field]) for record in records]
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 1:
        raise ValueError("batch-size and bootstrap-samples must be positive")
    if args.limit_images is not None and args.limit_images < 1:
        raise ValueError("limit-images must be positive")
    args.image_root = args.image_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.clip_checkpoint = args.clip_checkpoint.resolve()
    if not args.image_root.is_dir():
        raise FileNotFoundError(args.image_root)
    if not args.clip_checkpoint.is_file():
        raise FileNotFoundError(args.clip_checkpoint)

    records = collect_images(args.image_root, args.limit_images)
    if not records:
        raise ValueError("no generated PNG images found")
    prompt_ids = sorted({int(record["prompt_id"]) for record in records})
    relations = {
        relation: sum(record["relation"] == relation for record in records)
        for relation in RELATION_SWAP
    }
    discovery = {
        "status": "ok",
        "variant": args.variant,
        "images": len(records),
        "prompts": len(prompt_ids),
        "relations": relations,
        "image_root": str(args.image_root),
    }
    if args.dry_run:
        print(json.dumps(discovery, sort_keys=True))
        return 0
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("C7 CLIPScore requires CUDA")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    atomic_json(status_path, {**discovery, "status": "initializing"})
    started = time.perf_counter()
    try:
        device = torch.device(args.device)
        load_started = time.perf_counter()
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.clip_arch,
            pretrained=str(args.clip_checkpoint),
            device=device,
        )
        tokenizer = open_clip.get_tokenizer(args.clip_arch)
        model.eval().requires_grad_(False)
        model_load_seconds = time.perf_counter() - load_started

        unique_texts = sorted(
            {
                text
                for record in records
                for text in (record["prompt"], record["counterfactual_prompt"])
            }
        )
        text_features: dict[str, torch.Tensor] = {}
        with torch.inference_mode():
            for start in range(0, len(unique_texts), args.batch_size):
                texts = unique_texts[start : start + args.batch_size]
                tokens = tokenizer(texts).to(device)
                features = model.encode_text(tokens).float()
                features = features / features.norm(dim=-1, keepdim=True)
                for text, feature in zip(texts, features.cpu(), strict=True):
                    text_features[text] = feature

        score_started = time.perf_counter()
        scored: list[dict[str, Any]] = []
        with torch.inference_mode():
            for start in range(0, len(records), args.batch_size):
                batch = records[start : start + args.batch_size]
                images = []
                for record in batch:
                    with Image.open(record["image_path"]) as image:
                        images.append(preprocess(image.convert("RGB")))
                image_tensor = torch.stack(images).to(device)
                image_features = model.encode_image(image_tensor).float()
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                original_features = torch.stack(
                    [text_features[record["prompt"]] for record in batch]
                ).to(device)
                swapped_features = torch.stack(
                    [
                        text_features[record["counterfactual_prompt"]]
                        for record in batch
                    ]
                ).to(device)
                original_cosine = (image_features * original_features).sum(dim=-1)
                swapped_cosine = (image_features * swapped_features).sum(dim=-1)
                for record, original, swapped in zip(
                    batch, original_cosine.cpu(), swapped_cosine.cpu(), strict=True
                ):
                    raw = float(original)
                    counterfactual = float(swapped)
                    scored.append(
                        {
                            **record,
                            "variant": args.variant,
                            "raw_cosine": raw,
                            "clipscore": 2.5 * max(raw, 0.0),
                            "counterfactual_raw_cosine": counterfactual,
                            "counterfactual_clipscore": 2.5
                            * max(counterfactual, 0.0),
                            "relation_margin_raw": raw - counterfactual,
                            "relation_margin_clipscore": 2.5
                            * (max(raw, 0.0) - max(counterfactual, 0.0)),
                        }
                    )
                atomic_json(
                    status_path,
                    {
                        **discovery,
                        "status": "running",
                        "completed_images": len(scored),
                        "total_images": len(records),
                        "percent": 100.0 * len(scored) / len(records),
                    },
                )
        torch.cuda.synchronize(device)
        scoring_seconds = time.perf_counter() - score_started

        results_path = args.output_dir / "per_image_scores.jsonl"
        temporary = results_path.with_suffix(results_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in scored:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        os.replace(temporary, results_path)

        by_relation = {}
        for relation in RELATION_SWAP:
            subset = [record for record in scored if record["relation"] == relation]
            by_relation[relation] = {
                "images": len(subset),
                "clipscore": aggregate(subset, "clipscore"),
                "relation_margin_raw": aggregate(subset, "relation_margin_raw"),
            }
        positive_margins = sum(
            record["relation_margin_raw"] > 0.0 for record in scored
        )
        summary = {
            **discovery,
            "status": "completed",
            "protocol": "c7-clipscore-position72-v1",
            "clip_arch": args.clip_arch,
            "clip_checkpoint": str(args.clip_checkpoint),
            "clip_checkpoint_sha256": file_sha256(args.clip_checkpoint),
            "clipscore": aggregate(scored, "clipscore"),
            "raw_cosine": aggregate(scored, "raw_cosine"),
            "counterfactual_clipscore": aggregate(
                scored, "counterfactual_clipscore"
            ),
            "relation_margin_raw": {
                **aggregate(scored, "relation_margin_raw"),
                "prompt_cluster_bootstrap_95_ci": cluster_bootstrap_ci(
                    scored,
                    "relation_margin_raw",
                    args.bootstrap_samples,
                    args.bootstrap_seed,
                ),
                "positive_images": positive_margins,
                "positive_rate": positive_margins / len(scored),
            },
            "clipscore_prompt_cluster_bootstrap_95_ci": cluster_bootstrap_ci(
                scored,
                "clipscore",
                args.bootstrap_samples,
                args.bootstrap_seed,
            ),
            "by_relation": by_relation,
            "runtime": {
                "model_load_seconds": model_load_seconds,
                "scoring_seconds": scoring_seconds,
                "images_per_second": len(scored) / scoring_seconds,
                "total_seconds": time.perf_counter() - started,
                "batch_size": args.batch_size,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "open_clip": getattr(open_clip, "__version__", "unknown"),
                "device": torch.cuda.get_device_name(device),
            },
        }
        atomic_json(args.output_dir / "summary.json", summary)
        atomic_json(status_path, summary)
        print(json.dumps(summary, sort_keys=True))
        return 0
    except BaseException as error:
        atomic_json(
            status_path,
            {
                **discovery,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
