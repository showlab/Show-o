#!/usr/bin/env python3
"""Score paired Position-72 variants with the official ImageReward-v1.0 model."""

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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ImageReward as RM
import numpy as np
import torch
from PIL import Image
from transformers import BertTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Variant mapping in NAME=/absolute/path/to/geneval form.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/root/autodl-tmp/models/c7/ImageReward-v1.0/ImageReward.pt"),
    )
    parser.add_argument(
        "--med-config",
        type=Path,
        default=Path("/root/autodl-tmp/models/c7/ImageReward-v1.0/med_config.json"),
    )
    parser.add_argument(
        "--bert-tokenizer",
        type=Path,
        default=Path("/root/autodl-tmp/models/c7/bert-base-uncased-tokenizer"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-per-variant", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_variants(values: list[str]) -> dict[str, Path]:
    variants: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"variant must be NAME=PATH: {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in variants:
            raise ValueError(f"invalid or duplicate variant name: {name!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        variants[name] = path
    if len(variants) < 2:
        raise ValueError("at least two variants are required")
    return variants


def relation_from_metadata(metadata: dict[str, Any]) -> str:
    relations = [
        str(item["position"][0])
        for item in metadata.get("include", [])
        if "position" in item
    ]
    if len(relations) != 1:
        raise ValueError(f"expected one position relation, got {relations}")
    return relations[0]


def collect_variant(
    name: str, root: Path, limit: int | None
) -> dict[tuple[int, int], dict[str, Any]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for prompt_dir in sorted(root.iterdir()):
        if not prompt_dir.is_dir() or not prompt_dir.name.isdigit():
            continue
        metadata_path = prompt_dir / "metadata.jsonl"
        samples_dir = prompt_dir / "samples"
        if not metadata_path.is_file() or not samples_dir.is_dir():
            raise FileNotFoundError(f"incomplete GenEval directory: {prompt_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prompt = str(metadata["prompt"])
        relation = relation_from_metadata(metadata)
        for image_path in sorted(samples_dir.glob("*.png")):
            if not image_path.stem.isdigit():
                continue
            key = (int(prompt_dir.name), int(image_path.stem))
            records[key] = {
                "variant": name,
                "prompt_id": key[0],
                "sample_index": key[1],
                "prompt": prompt,
                "relation": relation,
                "image_path": str(image_path.resolve()),
            }
            if limit is not None and len(records) >= limit:
                return records
    return records


def validate_pairing(
    variants: dict[str, Path], limit: int | None
) -> tuple[list[str], dict[str, dict[tuple[int, int], dict[str, Any]]]]:
    names = list(variants)
    records = {
        name: collect_variant(name, root, limit) for name, root in variants.items()
    }
    reference_keys = set(records[names[0]])
    if not reference_keys:
        raise ValueError("no generated images found")
    for name in names[1:]:
        if set(records[name]) != reference_keys:
            raise ValueError(f"image keys do not match for {name}")
    for key in reference_keys:
        reference = records[names[0]][key]
        identity = (reference["prompt"], reference["relation"])
        for name in names[1:]:
            candidate = records[name][key]
            if (candidate["prompt"], candidate["relation"]) != identity:
                raise ValueError(f"prompt metadata mismatch for {name}: {key}")
    return names, records


def patch_local_tokenizer(path: Path) -> None:
    from ImageReward.models.BLIP import blip_pretrain

    def init_tokenizer() -> BertTokenizer:
        tokenizer = BertTokenizer.from_pretrained(str(path), local_files_only=True)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
        return tokenizer

    blip_pretrain.init_tokenizer = init_tokenizer


def score_batch(model: Any, batch: list[dict[str, Any]], device: torch.device) -> list[float]:
    images = []
    for record in batch:
        with Image.open(record["image_path"]) as image:
            images.append(model.preprocess(image.convert("RGB")))
    image_tensor = torch.stack(images).to(device)
    text_input = model.blip.tokenizer(
        [record["prompt"] for record in batch],
        padding="max_length",
        truncation=True,
        max_length=35,
        return_tensors="pt",
    ).to(device)
    image_embeds = model.blip.visual_encoder(image_tensor)
    image_atts = torch.ones(
        image_embeds.size()[:-1], dtype=torch.long, device=device
    )
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


def cluster_ci(
    values: dict[tuple[int, int], float], samples: int, seed: int
) -> list[float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for (prompt_id, _), value in values.items():
        grouped[prompt_id].append(value)
    means = np.asarray(
        [np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = rng.choice(means, size=(samples, len(means)), replace=True)
    return [float(value) for value in np.quantile(draws.mean(axis=1), [0.025, 0.975])]


def summarize(
    names: list[str],
    scored: dict[str, dict[tuple[int, int], dict[str, Any]]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {"variants": {}, "pairwise": {}}
    keys = sorted(scored[names[0]])
    for offset, name in enumerate(names):
        values = {key: float(scored[name][key]["image_reward"]) for key in keys}
        by_relation = {}
        relations = sorted({scored[name][key]["relation"] for key in keys})
        for relation in relations:
            subset = [
                values[key] for key in keys if scored[name][key]["relation"] == relation
            ]
            by_relation[relation] = {"images": len(subset), **aggregate(subset)}
        output["variants"][name] = {
            "images": len(values),
            "image_reward": aggregate(list(values.values())),
            "prompt_cluster_bootstrap_95_ci": cluster_ci(
                values, bootstrap_samples, bootstrap_seed + offset
            ),
            "by_relation": by_relation,
        }
    for offset, (left, right) in enumerate(itertools.combinations(names, 2)):
        deltas = {
            key: float(
                scored[right][key]["image_reward"]
                - scored[left][key]["image_reward"]
            )
            for key in keys
        }
        values = np.asarray(list(deltas.values()), dtype=np.float64)
        output["pairwise"][f"{left}__vs__{right}"] = {
            "delta_right_minus_left": float(values.mean()),
            "prompt_cluster_bootstrap_95_ci": cluster_ci(
                deltas, bootstrap_samples, bootstrap_seed + 100 + offset
            ),
            "right_wins": int(np.sum(values > 1e-12)),
            "left_wins": int(np.sum(values < -1e-12)),
            "ties": int(np.sum(np.abs(values) <= 1e-12)),
        }
    return output


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 1:
        raise ValueError("batch-size and bootstrap-samples must be positive")
    if args.limit_per_variant is not None and args.limit_per_variant < 1:
        raise ValueError("limit-per-variant must be positive")
    variants = parse_variants(args.variant)
    names, records = validate_pairing(variants, args.limit_per_variant)
    images_per_variant = len(records[names[0]])
    discovery = {
        "status": "ok",
        "variants": names,
        "images_per_variant": images_per_variant,
        "total_images": images_per_variant * len(names),
        "prompts": len({key[0] for key in records[names[0]]}),
    }
    if args.dry_run:
        print(json.dumps(discovery, sort_keys=True))
        return 0
    checkpoint = args.checkpoint.resolve()
    med_config = args.med_config.resolve()
    bert_tokenizer = args.bert_tokenizer.resolve()
    for path in (checkpoint, med_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not bert_tokenizer.is_dir():
        raise FileNotFoundError(bert_tokenizer)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ImageReward C7 scoring requires CUDA")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    atomic_json(status_path, {**discovery, "status": "loading_model", "updated_at": now()})
    started = time.perf_counter()
    try:
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cuda")
        patch_local_tokenizer(bert_tokenizer)
        load_started = time.perf_counter()
        model = RM.load(
            str(checkpoint),
            device=device,
            med_config=str(med_config),
        )
        model.eval().requires_grad_(False)
        model_load_seconds = time.perf_counter() - load_started
        torch.cuda.reset_peak_memory_stats(device)

        ordered = [
            records[name][key]
            for key in sorted(records[names[0]])
            for name in names
        ]
        scored_flat: list[dict[str, Any]] = []
        score_started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(ordered), args.batch_size):
                batch = ordered[start : start + args.batch_size]
                rewards = score_batch(model, batch, device)
                for record, reward in zip(batch, rewards, strict=True):
                    scored_flat.append({**record, "image_reward": reward})
                atomic_json(
                    status_path,
                    {
                        **discovery,
                        "status": "running",
                        "completed_images": len(scored_flat),
                        "percent": 100.0 * len(scored_flat) / len(ordered),
                        "updated_at": now(),
                    },
                )
        torch.cuda.synchronize(device)
        scoring_seconds = time.perf_counter() - score_started
        atomic_jsonl(output_dir / "per_image_scores.jsonl", scored_flat)
        scored: dict[str, dict[tuple[int, int], dict[str, Any]]] = {
            name: {} for name in names
        }
        for record in scored_flat:
            key = (int(record["prompt_id"]), int(record["sample_index"]))
            scored[record["variant"]][key] = record
        comparison = summarize(
            names,
            scored,
            args.bootstrap_samples,
            args.bootstrap_seed,
        )
        summary = {
            **discovery,
            **comparison,
            "status": "completed",
            "protocol": "c7-imagereward-position72-paired-v1",
            "model": "ImageReward-v1.0",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "med_config": str(med_config),
            "bert_tokenizer": str(bert_tokenizer),
            "runtime": {
                "model_load_seconds": model_load_seconds,
                "scoring_seconds": scoring_seconds,
                "images_per_second": len(scored_flat) / scoring_seconds,
                "total_seconds": time.perf_counter() - started,
                "batch_size": args.batch_size,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(device),
            },
        }
        atomic_json(output_dir / "summary.json", summary)
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
                "updated_at": now(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
