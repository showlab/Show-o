#!/usr/bin/env python3
"""Evaluate baseline and DRSA checkpoints on the held-out VSR zero-shot test."""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from functools import partial
from pathlib import Path

import torch

from datasets.drsa_dataset import VsrRelationDataset
from datasets.utils import image_transform
from evaluation.c3_layout_guidance.run_quick8_v2 import (
    load_drsa_checkpoint,
    utc_now,
    write_json,
)
from evaluation.c4_drsa_adapter.train_drsa import build_understanding_sequence
from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
from models.misc import get_text_tokenizer
from utils import path_to_llm_name


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VARIANTS = ROOT / "evaluation/c4_drsa_adapter/drsa_sweep_variants.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant-config-json", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--variants", default="baseline,drsa_0500,drsa_1000,drsa_1500,drsa_2000,drsa_2500,drsa_3000")
    parser.add_argument("--model", default="showlab/show-o2-1.5B")
    parser.add_argument("--vae-path", type=Path, default=ROOT / "Wan2.1_VAE.pth")
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/data/drsa"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def variant_paths(path: Path, names: tuple[str, ...]) -> dict[str, str | None]:
    configs = json.loads(path.read_text())
    result = {}
    for name in names:
        if name == "baseline":
            result[name] = None
        else:
            checkpoint = configs.get(name, {}).get("drsa_checkpoint")
            if not checkpoint:
                raise KeyError(f"No drsa_checkpoint configured for {name}")
            result[name] = str(Path(checkpoint).expanduser().resolve())
    return result


def atomic_json(path: Path, payload: dict) -> None:
    write_json(path, payload)


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    if not variants or variants[0] != "baseline" or len(set(variants)) != len(variants):
        raise ValueError("variants must be unique and start with baseline")
    checkpoints = variant_paths(args.variant_config_json, variants)
    annotation = args.data_root / "vsr/annotations/zeroshot/test.jsonl"
    image_zip = args.data_root / "vsr/vsr_images.zip"
    dataset = VsrRelationDataset(
        annotation,
        image_zip,
        image_transform=partial(image_transform, resolution=432),
    )
    total_rows = len(dataset) if args.limit is None else min(len(dataset), args.limit)
    if total_rows < 1:
        raise ValueError("VSR evaluation contains no rows")
    total_predictions = total_rows * len(variants)
    status_path = args.run_dir / "status.json"
    started = time.time()
    status = {
        "status": "initializing",
        "phase": "loading_model",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "completed_predictions": 0,
        "total_predictions": total_predictions,
        "percent": 0.0,
    }
    atomic_json(status_path, status)

    device = torch.device(args.device)
    dtype = torch.bfloat16
    llm_path = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer, token_ids = get_text_tokenizer(
        llm_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[llm_path],
    )
    yes_ids = tokenizer(" yes", add_special_tokens=False)["input_ids"]
    no_ids = tokenizer(" no", add_special_tokens=False)["input_ids"]
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise RuntimeError(f"VSR labels are not single tokens: yes={yes_ids}, no={no_ids}")
    model = Showo2Qwen2_5.from_pretrained(
        args.model,
        use_safetensors=False,
        local_files_only=True,
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
        drsa_enabled=True,
    ).to(device=device, dtype=dtype).eval().requires_grad_(False)
    vae = WanVAE(vae_pth=str(args.vae_path), dtype=dtype, device=device)
    sequence_length = args.max_text_tokens + 734

    completed = 0
    summaries = {}
    for variant in variants:
        metadata = load_drsa_checkpoint(model, checkpoints[variant])
        variant_dir = args.run_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(variant_dir / "drsa_checkpoint.json", metadata)
        results_path = variant_dir / "results.jsonl"
        existing = read_jsonl(results_path)
        if existing and len(existing) != total_rows:
            raise RuntimeError(f"Partial VSR result cannot be resumed safely: {results_path}")
        if existing:
            correct = sum(bool(row["correct"]) for row in existing)
            completed += len(existing)
            summaries[variant] = {"correct": correct, "total": len(existing), "accuracy": correct / len(existing)}
            continue

        correct = 0
        with results_path.open("w", encoding="utf-8") as output:
            for index in range(total_rows):
                row = dataset[index]
                image = row.pop("image").unsqueeze(0).to(device)
                with torch.no_grad():
                    latents = vae.sample(
                        image.to(dtype=dtype).unsqueeze(2), deterministic=True
                    ).squeeze(2).to(dtype)
                sequence = build_understanding_sequence(
                    row, tokenizer, token_ids, args.max_text_tokens, sequence_length
                )
                text_tokens, text_labels, positions, text_mask, image_mask = [
                    value.unsqueeze(0).to(device) for value in sequence
                ]
                answer_start = int((text_labels[0] != -100).nonzero()[0])
                attention_mask = omni_attn_mask_naive(
                    1, sequence_length, positions, device
                ).to(dtype)
                relation_code = None
                if metadata["enabled"]:
                    encoded = tokenizer(
                        [str(row["relation"])], add_special_tokens=False,
                        truncation=True, max_length=32, return_tensors="pt",
                    )
                    relation_tokens = encoded["input_ids"].to(device)
                    relation_mask = encoded["attention_mask"].to(device)
                    features = model.showo.model.embed_tokens(relation_tokens)
                    relation_code = model.drsa_adapter.encode_relation(features, relation_mask)
                with torch.no_grad():
                    logits = model.forward_und_only(
                        text_tokens=text_tokens,
                        image_latents=latents,
                        t=torch.ones((1,), device=device, dtype=dtype),
                        attention_mask=attention_mask,
                        text_masks=text_mask,
                        image_masks=image_mask,
                        modality_positions=positions,
                        output_hidden_states=True,
                        max_seq_len=sequence_length,
                        device=device,
                        drsa_relation_codes=relation_code,
                    )
                scores = logits[0, answer_start - 1, [yes_ids[0], no_ids[0]]].float()
                prediction = 1 if int(scores.argmax()) == 0 else 0
                is_correct = prediction == int(row["label"])
                correct += int(is_correct)
                record = {
                    "index": index,
                    "relation": str(row["relation"]),
                    "label": int(row["label"]),
                    "prediction": prediction,
                    "correct": is_correct,
                    "yes_logit": float(scores[0]),
                    "no_logit": float(scores[1]),
                }
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                completed += 1
                elapsed = time.time() - started
                eta = elapsed / completed * (total_predictions - completed)
                status.update({
                    "status": "running",
                    "phase": "vsr_test",
                    "variant": variant,
                    "variant_completed": index + 1,
                    "variant_total": total_rows,
                    "completed_predictions": completed,
                    "total_predictions": total_predictions,
                    "percent": 100.0 * completed / total_predictions,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "running_correct": correct,
                    "updated_at": utc_now(),
                })
                atomic_json(status_path, status)
        summaries[variant] = {"correct": correct, "total": total_rows, "accuracy": correct / total_rows}
        atomic_json(variant_dir / "summary.json", summaries[variant])

    status.update({
        "status": "completed",
        "phase": "completed",
        "percent": 100.0,
        "completed_predictions": total_predictions,
        "summaries": summaries,
        "updated_at": utc_now(),
    })
    atomic_json(status_path, status)
    atomic_json(args.run_dir / "summary.json", {"status": "completed", "variants": summaries})
    print(json.dumps({"status": "completed", "variants": summaries}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        parsed = parse_args()
        parsed.run_dir.mkdir(parents=True, exist_ok=True)
        write_json(parsed.run_dir / "status.json", {
            "status": "failed",
            "phase": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "updated_at": utc_now(),
        })
        raise
