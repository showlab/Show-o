#!/usr/bin/env python3
"""Measure frozen-U teacher validity on real COCO images for TIFA questions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.utils import image_transform
from evaluation.c4_drsa_adapter.train_drsa import encode_images
from evaluation.c6_mask2former_alignment.frozen_u_cycle import (
    frozen_u_candidate_logits,
)
from evaluation.c6_mask2former_alignment.tifa_cycle import (
    TifaCocoCycleDataset,
    build_tifa_question_batch,
    option_token_ids,
)
from evaluation.c6_mask2former_alignment.tifa_cycle_sampling import (
    prepare_cycle_questions,
)
from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
from models.misc import get_text_tokenizer
from utils import path_to_llm_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-questions", type=int, default=4)
    parser.add_argument("--max-text-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--gate-margin", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default="showlab/show-o2-1.5B")
    parser.add_argument("--vae-path", type=Path, default=ROOT / "Wan2.1_VAE.pth")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("TIFA teacher gate audit requires CUDA")
    device, dtype = torch.device(args.device), torch.bfloat16
    dataset = TifaCocoCycleDataset(
        args.manifest,
        image_transform=lambda image: image_transform(image, resolution=432),
    )
    total_rows = len(dataset) if args.limit is None else min(len(dataset), args.limit)
    llm_path = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer, token_ids = get_text_tokenizer(
        llm_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[llm_path],
    )
    candidate_ids = option_token_ids(tokenizer)
    sequence_length = args.max_text_tokens + 734
    model = Showo2Qwen2_5.from_pretrained(
        args.model,
        use_safetensors=False,
        local_files_only=True,
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
        drsa_enabled=False,
        generation_adapter_enabled=False,
    )
    model.to(device=device, dtype=dtype).requires_grad_(False).eval()
    vae = WanVAE(vae_pth=str(args.vae_path), dtype=dtype, device=device)
    vae.model.requires_grad_(False).eval()

    question_total = 0
    question_valid = 0
    prompt_any = 0
    prompt_two_questions = 0
    prompt_two_dimensions = 0
    prompt_all = 0
    dimension_total: Counter[str] = Counter()
    dimension_valid: Counter[str] = Counter()
    target_total: Counter[int] = Counter()
    prediction_total: Counter[int] = Counter()
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    details_path = args.output.with_suffix(".questions.jsonl")
    with details_path.open("w", encoding="utf-8") as detail_handle:
        for index in range(total_rows):
            row = dataset[index]
            selected = prepare_cycle_questions(
                row,
                max_questions=args.max_questions,
                base_seed=args.seed,
                cycle_step=index + 1,
            )
            batch = build_tifa_question_batch(
                selected,
                tokenizer,
                token_ids,
                max_text_tokens=args.max_text_tokens,
                sequence_length=sequence_length,
                device=device,
            )
            with torch.no_grad():
                image_latent = encode_images(
                    vae, row["image"].unsqueeze(0), dtype
                )
                logits = frozen_u_candidate_logits(
                    model,
                    image_latent,
                    batch,
                    candidate_ids,
                    sequence_length=sequence_length,
                    dtype=dtype,
                    device=device,
                    attention_mask_builder=omni_attn_mask_naive,
                    relation_codes=None,
                ).float()
            mask = batch.candidate_mask
            masked = logits.masked_fill(~mask, float("-inf"))
            indices = torch.arange(len(batch.questions), device=device)
            target_logits = masked[indices, batch.targets]
            competitor_logits = masked.masked_fill(
                F.one_hot(batch.targets, num_classes=masked.shape[-1]).bool(),
                float("-inf"),
            ).amax(dim=-1)
            margins = target_logits - competitor_logits
            predictions = masked.argmax(dim=-1)
            valid = predictions.eq(batch.targets) & (margins > args.gate_margin)
            valid_dimensions = {
                question.dimension
                for question, accepted in zip(batch.questions, valid.tolist())
                if accepted
            }
            valid_count = int(valid.sum().cpu())
            question_total += len(batch.questions)
            question_valid += valid_count
            prompt_any += int(valid_count >= 1)
            prompt_two_questions += int(valid_count >= 2)
            prompt_two_dimensions += int(len(valid_dimensions) >= 2)
            prompt_all += int(valid_count == len(batch.questions))
            for question, target, prediction, margin, accepted in zip(
                batch.questions,
                batch.targets.tolist(),
                predictions.tolist(),
                margins.tolist(),
                valid.tolist(),
            ):
                dimension_total[question.dimension] += 1
                dimension_valid[question.dimension] += int(accepted)
                target_total[int(target)] += 1
                prediction_total[int(prediction)] += 1
                detail_handle.write(json.dumps({
                    "sample_id": row["sample_id"],
                    "dimension": question.dimension,
                    "question": question.question,
                    "choices": list(question.choices),
                    "target": int(target),
                    "prediction": int(prediction),
                    "target_margin": float(margin),
                    "teacher_valid": bool(accepted),
                }, sort_keys=True) + "\n")
            rows.append({
                "sample_id": row["sample_id"],
                "questions": len(batch.questions),
                "valid_questions": valid_count,
                "valid_dimensions": sorted(valid_dimensions),
            })
            if (index + 1) % 10 == 0 or index + 1 == total_rows:
                print(
                    f"prompts={index+1}/{total_rows} "
                    f"question_valid={question_valid}/{question_total} "
                    f"two_dimension_prompts={prompt_two_dimensions}",
                    flush=True,
                )

    report = {
        "status": "completed",
        "protocol": "frozen-u-real-coco-per-question-gate-v1",
        "manifest": str(args.manifest.resolve()),
        "prompts": total_rows,
        "questions": question_total,
        "gate_margin": args.gate_margin,
        "valid_questions": question_valid,
        "valid_question_fraction": question_valid / question_total,
        "prompts_with_any_valid": prompt_any,
        "prompts_with_any_valid_fraction": prompt_any / total_rows,
        "prompts_with_two_valid_questions": prompt_two_questions,
        "prompts_with_two_valid_questions_fraction": prompt_two_questions / total_rows,
        "prompts_with_two_valid_dimensions": prompt_two_dimensions,
        "prompts_with_two_valid_dimensions_fraction": prompt_two_dimensions / total_rows,
        "prompts_all_questions_valid": prompt_all,
        "prompts_all_questions_valid_fraction": prompt_all / total_rows,
        "by_dimension": {
            name: {
                "valid": dimension_valid[name],
                "total": total,
                "fraction": dimension_valid[name] / total,
            }
            for name, total in sorted(dimension_total.items())
        },
        "target_counts": dict(sorted(target_total.items())),
        "prediction_counts": dict(sorted(prediction_total.items())),
        "details": str(details_path.resolve()),
        "prompt_rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: value for key, value in report.items()
        if key not in ("prompt_rows", "by_dimension")
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
