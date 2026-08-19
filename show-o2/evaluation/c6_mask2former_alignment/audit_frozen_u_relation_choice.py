#!/usr/bin/env python3
"""Audit frozen Show-o2 U routes as a four-way spatial relation readout."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.drsa_dataset import BprimeCocoBoxDataset
from datasets.utils import image_transform
from evaluation.c4_drsa_adapter.train_drsa import (
    atomic_json,
    encode_images,
    now_iso,
    relation_code,
)
from evaluation.c5_boxdiff_adapter.train_bprime_position import load_b1_checkpoint
from evaluation.c6_mask2former_alignment.frozen_u_cycle import (
    RELATION_OPTIONS,
    build_relation_choice_batch,
    frozen_u_candidate_logits,
    relation_option_token_ids,
)
from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
from models.misc import get_text_tokenizer
from utils import path_to_llm_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=WORKSPACE / "data/drsa/mask2former_alignment_5k/train.jsonl",
    )
    parser.add_argument(
        "--shard-root",
        type=Path,
        default=WORKSPACE / "data/drsa/spright_coco/data",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=WORKSPACE / "runs/showo2_bprime/bprime-full-v2-20260808T030439Z/b1/checkpoints/step-000200.pt",
    )
    parser.add_argument("--model", default="showlab/show-o2-1.5B")
    parser.add_argument("--vae-path", type=Path, default=ROOT / "Wan2.1_VAE.pth")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def summarize(rows: list[dict[str, Any]], route: str) -> dict[str, Any]:
    selected = [row for row in rows if row["route"] == route]
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    correct = 0
    positive_margin = 0
    for row in selected:
        confusion[row["target_relation"]][row["prediction_relation"]] += 1
        correct += int(row["correct"])
        positive_margin += int(row["target_margin"] > 0.0)
    return {
        "samples": len(selected),
        "correct": correct,
        "accuracy": correct / len(selected) if selected else None,
        "positive_target_margin": positive_margin,
        "positive_target_margin_rate": positive_margin / len(selected) if selected else None,
        "prediction_counts": dict(Counter(row["prediction_relation"] for row in selected)),
        "confusion": {
            relation: dict(confusion[relation]) for relation in RELATION_OPTIONS
        },
        "mean_target_margin": (
            sum(float(row["target_margin"]) for row in selected) / len(selected)
            if selected else None
        ),
    }


def main() -> int:
    args = parse_args()
    if args.limit < 1 or args.max_text_tokens < 1:
        raise ValueError("limit and max-text-tokens must be positive")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("frozen-U relation audit requires CUDA")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.run_dir / "results.jsonl"
    if results_path.exists():
        raise FileExistsError(results_path)

    device, dtype = torch.device(args.device), torch.bfloat16
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer, token_ids = get_text_tokenizer(
        "Qwen/Qwen2.5-1.5B-Instruct",
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name["Qwen/Qwen2.5-1.5B-Instruct"],
    )
    candidate_ids = relation_option_token_ids(tokenizer)
    sequence_length = args.max_text_tokens + 734
    model = Showo2Qwen2_5.from_pretrained(
        args.model,
        use_safetensors=False,
        local_files_only=True,
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
        drsa_enabled=True,
    ).to(device=device, dtype=dtype).eval().requires_grad_(False)
    checkpoint_step = load_b1_checkpoint(model, args.init_checkpoint)
    vae = WanVAE(vae_pth=str(args.vae_path), dtype=dtype, device=device)
    vae.model.eval().requires_grad_(False)
    dataset = BprimeCocoBoxDataset(
        args.manifest,
        args.shard_root,
        image_transform=partial(image_transform, resolution=432),
    )
    order = list(range(len(dataset)))
    random.Random(args.seed).shuffle(order)
    order = order[: min(args.limit, len(order))]
    protocol = {
        "protocol": "frozen-u-relation-choice-audit-v1",
        "started_at": now_iso(),
        "manifest": str(args.manifest.resolve()),
        "init_checkpoint": str(args.init_checkpoint.resolve()),
        "init_checkpoint_step": checkpoint_step,
        "seed": args.seed,
        "limit": len(order),
        "relation_options": list(RELATION_OPTIONS),
        "candidate_token_ids": list(candidate_ids),
        "routes": ["base_u", "drsa_u_question_conditioned"],
        "answer_image_only": True,
    }
    atomic_json(args.run_dir / "protocol.json", protocol)

    rows = []
    with torch.no_grad():
        for position, index in enumerate(order, 1):
            row = dataset[index]
            phrases = [str(item["phrase"]) for item in row["objects"]]
            relation = str(row["relation"])
            image_latent = encode_images(vae, row["image"].unsqueeze(0), dtype)
            batch = build_relation_choice_batch(
                phrases,
                relation,
                tokenizer,
                token_ids,
                max_text_tokens=args.max_text_tokens,
                sequence_length=sequence_length,
                device=device,
            )
            question_code = relation_code(
                model, tokenizer, [batch.questions[0].statement], device
            )
            for route, code in (
                ("base_u", None),
                ("drsa_u_question_conditioned", question_code),
            ):
                logits = frozen_u_candidate_logits(
                    model,
                    image_latent,
                    batch,
                    candidate_ids,
                    sequence_length=sequence_length,
                    dtype=dtype,
                    device=device,
                    attention_mask_builder=omni_attn_mask_naive,
                    relation_codes=code,
                )[0]
                target = int(batch.targets[0])
                prediction = int(logits.argmax())
                competing = torch.cat((logits[:target], logits[target + 1 :])).max()
                record = {
                    "position": position,
                    "dataset_index": index,
                    "sample_id": str(row["sample_id"]),
                    "phrases": phrases,
                    "target_relation": relation,
                    "target_index": target,
                    "route": route,
                    "logits": logits.cpu().tolist(),
                    "probabilities": logits.softmax(dim=-1).cpu().tolist(),
                    "prediction_relation": RELATION_OPTIONS[prediction],
                    "prediction_index": prediction,
                    "correct": prediction == target,
                    "target_margin": float((logits[target] - competing).cpu()),
                }
                append_jsonl(results_path, record)
                rows.append(record)
            print(f"audit={position}/{len(order)} relation={relation}", flush=True)

    summary = {
        **protocol,
        "status": "completed",
        "completed_at": now_iso(),
        "metrics": {
            route: summarize(rows, route)
            for route in ("base_u", "drsa_u_question_conditioned")
        },
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    atomic_json(args.run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
