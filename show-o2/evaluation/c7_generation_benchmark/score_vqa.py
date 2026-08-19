#!/usr/bin/env python3
"""Score paired Position variants with Qwen2.5-VL VQAScore.

The metric follows t2v-metrics' Qwen implementation: VQAScore is the
next-token probability of ``Yes`` for ``Does this image show \"<text>\"?``.
Each image is scored against both its original prompt and the prompt with the
spatial relation swapped, so the output also contains a counterfactual margin.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


RELATION_SWAP = {
    "left of": "right of",
    "right of": "left of",
    "above": "below",
    "below": "above",
}
PROTOCOL = "c7-vqascore-qwen2.5-vl-7b-position72-paired-v1"


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
        "--model-path",
        type=Path,
        default=Path("/root/autodl-tmp/models/c7/Qwen2.5-VL-7B-Instruct"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-per-variant", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
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


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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
    if not variants:
        raise ValueError("at least one variant is required")
    return variants


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
            f"prompt must contain relation exactly once: relation={relation!r} "
            f"prompt={prompt!r}"
        )
    return prompt.replace(relation, RELATION_SWAP[relation], 1)


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
        swapped = counterfactual_prompt(prompt, relation)
        for image_path in sorted(samples_dir.glob("*.png")):
            if not image_path.stem.isdigit():
                continue
            key = (int(prompt_dir.name), int(image_path.stem))
            records[key] = {
                "variant": name,
                "prompt_id": key[0],
                "sample_index": key[1],
                "prompt": prompt,
                "counterfactual_prompt": swapped,
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
        identity = (
            reference["prompt"],
            reference["counterfactual_prompt"],
            reference["relation"],
        )
        for name in names[1:]:
            candidate = records[name][key]
            if (
                candidate["prompt"],
                candidate["counterfactual_prompt"],
                candidate["relation"],
            ) != identity:
                raise ValueError(f"prompt metadata mismatch for {name}: {key}")
    return names, records


def result_key(record: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(record["variant"]),
        int(record["prompt_id"]),
        int(record["sample_index"]),
    )


def load_partial(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    if not path.is_file():
        return {}
    scored: dict[tuple[str, int, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = result_key(record)
            if key in scored:
                raise ValueError(f"duplicate partial score at line {line_number}: {key}")
            if record.get("protocol") != PROTOCOL:
                raise ValueError(f"partial score protocol mismatch at line {line_number}")
            scored[key] = record
    return scored


def question(text: str) -> str:
    return f'Does this image show "{text}"?'


def score_questions(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Any,
    batch: list[dict[str, Any]],
    text_field: str,
    yes_token_id: int,
    no_token_id: int,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    prompts: list[str] = []
    images: list[Image.Image] = []
    try:
        for record in batch:
            image = Image.open(record["image_path"]).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question(str(record[text_field]))},
                    ],
                }
            ]
            prompts.append(
                processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            image_inputs, video_inputs = process_vision_info(messages)
            if video_inputs:
                raise ValueError("image VQAScore unexpectedly produced video inputs")
            if len(image_inputs) != 1:
                raise ValueError("each VQAScore record must produce exactly one image")
            images.extend(image_inputs)

        inputs = processor(
            text=prompts,
            images=images,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        probabilities = torch.softmax(outputs.scores[0].float(), dim=-1)
        yes = probabilities[:, yes_token_id].detach().cpu().tolist()
        no = probabilities[:, no_token_id].detach().cpu().tolist()
        return [float(value) for value in yes], [float(value) for value in no]
    finally:
        for image in images:
            image.close()


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
    scored: dict[tuple[str, int, int], dict[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {"variants": {}, "pairwise": {}}
    per_variant = {
        name: {
            (key[1], key[2]): record
            for key, record in scored.items()
            if key[0] == name
        }
        for name in names
    }
    reference_keys = sorted(per_variant[names[0]])
    for name in names[1:]:
        if sorted(per_variant[name]) != reference_keys:
            raise ValueError(f"scored result keys do not match for {name}")

    for offset, name in enumerate(names):
        rows = per_variant[name]
        full = {key: float(rows[key]["vqascore"]) for key in reference_keys}
        swapped = {
            key: float(rows[key]["counterfactual_vqascore"])
            for key in reference_keys
        }
        margins = {
            key: full[key] - swapped[key] for key in reference_keys
        }
        relations = sorted({str(rows[key]["relation"]) for key in reference_keys})
        by_relation = {}
        for relation in relations:
            selected = [
                key for key in reference_keys if rows[key]["relation"] == relation
            ]
            by_relation[relation] = {
                "images": len(selected),
                "vqascore": aggregate([full[key] for key in selected]),
                "counterfactual_vqascore": aggregate(
                    [swapped[key] for key in selected]
                ),
                "relation_margin": aggregate([margins[key] for key in selected]),
            }
        positive = sum(value > 0.0 for value in margins.values())
        output["variants"][name] = {
            "images": len(reference_keys),
            "vqascore": aggregate(list(full.values())),
            "vqascore_prompt_cluster_bootstrap_95_ci": cluster_ci(
                full, bootstrap_samples, bootstrap_seed + offset
            ),
            "counterfactual_vqascore": aggregate(list(swapped.values())),
            "relation_margin": {
                **aggregate(list(margins.values())),
                "positive_images": positive,
                "positive_rate": positive / len(margins),
                "prompt_cluster_bootstrap_95_ci": cluster_ci(
                    margins, bootstrap_samples, bootstrap_seed + 20 + offset
                ),
            },
            "by_relation": by_relation,
        }

    for offset, (left, right) in enumerate(itertools.combinations(names, 2)):
        left_rows, right_rows = per_variant[left], per_variant[right]
        deltas = {
            key: float(right_rows[key]["vqascore"] - left_rows[key]["vqascore"])
            for key in reference_keys
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
    names, source_records = validate_pairing(variants, args.limit_per_variant)
    images_per_variant = len(source_records[names[0]])
    total_images = images_per_variant * len(names)
    discovery = {
        "status": "ok",
        "protocol": PROTOCOL,
        "variants": names,
        "images_per_variant": images_per_variant,
        "total_images": total_images,
        "total_vqa_queries": total_images * 2,
        "prompts": len({key[0] for key in source_records[names[0]]}),
    }
    if args.dry_run:
        print(json.dumps(discovery, sort_keys=True))
        return 0
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Qwen VQAScore requires CUDA")

    model_path = args.model_path.expanduser().resolve()
    if not (model_path / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(model_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    results_path = output_dir / "per_image_scores.jsonl"
    if args.no_resume and results_path.exists():
        raise FileExistsError(
            f"refusing to overwrite partial results with --no-resume: {results_path}"
        )
    scored = {} if args.no_resume else load_partial(results_path)
    expected = {
        (name, key[0], key[1])
        for name in names
        for key in source_records[name]
    }
    unexpected = set(scored) - expected
    if unexpected:
        raise ValueError(f"partial results contain unexpected keys: {sorted(unexpected)[:3]}")

    ordered = [
        source_records[name][key]
        for key in sorted(source_records[names[0]])
        for name in names
    ]
    remaining = [record for record in ordered if result_key(record) not in scored]
    initial_completed = len(scored)
    started = time.perf_counter()
    atomic_json(
        status_path,
        {
            **discovery,
            "status": "loading_model" if remaining else "aggregating",
            "completed_images": initial_completed,
            "completed_vqa_queries": initial_completed * 2,
            "percent": 100.0 * initial_completed / total_images,
            "updated_at": now(),
        },
    )

    model_load_seconds = 0.0
    scoring_seconds = 0.0
    try:
        if remaining:
            torch.manual_seed(0)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            load_started = time.perf_counter()
            processor = AutoProcessor.from_pretrained(
                str(model_path), local_files_only=True, use_fast=False
            )
            processor.tokenizer.padding_side = "left"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16,
                attn_implementation=args.attn_implementation,
                device_map="auto",
                local_files_only=True,
            )
            model.eval().requires_grad_(False)
            device = next(model.parameters()).device
            yes_tokens = processor.tokenizer.encode("Yes")
            no_tokens = processor.tokenizer.encode("No")
            if len(yes_tokens) != 1 or len(no_tokens) != 1:
                raise ValueError(
                    f"Yes/No must each be one token: Yes={yes_tokens}, No={no_tokens}"
                )
            yes_token_id, no_token_id = yes_tokens[0], no_tokens[0]
            model_load_seconds = time.perf_counter() - load_started
            torch.cuda.reset_peak_memory_stats(device)
            score_started = time.perf_counter()
            for start in range(0, len(remaining), args.batch_size):
                batch = remaining[start : start + args.batch_size]
                yes, no = score_questions(
                    model,
                    processor,
                    batch,
                    "prompt",
                    yes_token_id,
                    no_token_id,
                    device,
                )
                swapped_yes, swapped_no = score_questions(
                    model,
                    processor,
                    batch,
                    "counterfactual_prompt",
                    yes_token_id,
                    no_token_id,
                    device,
                )
                completed_batch = []
                for record, full_yes, full_no, cf_yes, cf_no in zip(
                    batch, yes, no, swapped_yes, swapped_no, strict=True
                ):
                    completed_batch.append(
                        {
                            **record,
                            "protocol": PROTOCOL,
                            "question": question(record["prompt"]),
                            "counterfactual_question": question(
                                record["counterfactual_prompt"]
                            ),
                            "vqascore": full_yes,
                            "no_probability": full_no,
                            "counterfactual_vqascore": cf_yes,
                            "counterfactual_no_probability": cf_no,
                            "relation_margin": full_yes - cf_yes,
                        }
                    )
                append_jsonl(results_path, completed_batch)
                for record in completed_batch:
                    scored[result_key(record)] = record
                run_elapsed = time.perf_counter() - score_started
                completed_this_run = len(scored) - initial_completed
                rate = completed_this_run / run_elapsed
                remaining_images = total_images - len(scored)
                by_variant = {
                    name: sum(key[0] == name for key in scored) for name in names
                }
                atomic_json(
                    status_path,
                    {
                        **discovery,
                        "status": "running",
                        "phase": "original_and_counterfactual_vqascore",
                        "completed_images": len(scored),
                        "completed_vqa_queries": len(scored) * 2,
                        "completed_by_variant": by_variant,
                        "percent": 100.0 * len(scored) / total_images,
                        "images_per_second": rate,
                        "eta_seconds": remaining_images / max(rate, 1e-12),
                        "model_load_seconds": model_load_seconds,
                        "updated_at": now(),
                    },
                )
            torch.cuda.synchronize(device)
            scoring_seconds = time.perf_counter() - score_started
            peak_memory = torch.cuda.max_memory_allocated(device)
        else:
            yes_token_id = no_token_id = None
            peak_memory = 0

        atomic_json(
            status_path,
            {
                **discovery,
                "status": "aggregating",
                "completed_images": len(scored),
                "completed_vqa_queries": len(scored) * 2,
                "percent": 100.0,
                "updated_at": now(),
            },
        )
        aggregate_output = summarize(
            names, scored, args.bootstrap_samples, args.bootstrap_seed
        )
        summary = {
            **discovery,
            **aggregate_output,
            "status": "completed",
            "model": "Qwen2.5-VL-7B-Instruct",
            "model_path": str(model_path),
            "model_index_sha256": sha256(
                model_path / "model.safetensors.index.json"
            ),
            "question_template": 'Does this image show "{}"?',
            "answer_token": "Yes",
            "yes_token_id": yes_token_id,
            "no_token_id": no_token_id,
            "attn_implementation": args.attn_implementation,
            "implementation": {
                "formula": "P(Yes as next token | image, question)",
                "reference": "t2v-metrics qwen2vl_model.py",
                "t2v_metrics_version": importlib.metadata.version("t2v-metrics"),
                "note": "Standalone image-only runner avoids optional video/API eager imports.",
            },
            "runtime": {
                "batch_size": args.batch_size,
                "model_load_seconds": model_load_seconds,
                "scoring_seconds_this_run": scoring_seconds,
                "scored_images_this_run": len(scored) - initial_completed,
                "total_seconds_this_run": time.perf_counter() - started,
                "peak_cuda_memory_bytes": peak_memory,
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": importlib.metadata.version("transformers"),
                "device": torch.cuda.get_device_name(0),
            },
            "updated_at": now(),
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
                "completed_images": len(scored),
                "completed_vqa_queries": len(scored) * 2,
                "percent": 100.0 * len(scored) / total_images,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "updated_at": now(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
