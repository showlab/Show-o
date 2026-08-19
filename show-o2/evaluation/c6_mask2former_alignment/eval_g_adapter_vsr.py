#!/usr/bin/env python3
"""Evaluate G-adapter checkpoints on held-out VSR with an explicit U bridge."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.drsa_dataset import VsrRelationDataset
from datasets.utils import image_transform
from evaluation.c4_drsa_adapter.train_drsa import build_understanding_sequence
from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
from models.generation_adaptation import checkpoint_model_kwargs, configure_from_checkpoint
from models.misc import get_text_tokenizer
from utils import path_to_llm_name


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--bridge", choices=("on", "off"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/data/drsa"))
    parser.add_argument("--model", default="showlab/show-o2-1.5B")
    parser.add_argument("--vae-path", type=Path, default=ROOT / "Wan2.1_VAE.pth")
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[bool]] = defaultdict(list)
    labels: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        groups[str(row["relation"])].append(bool(row["correct"]))
        labels[int(row["label"])].append(bool(row["correct"]))
    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows),
        "by_label": {
            str(label): {"correct": sum(values), "total": len(values), "accuracy": sum(values) / len(values)}
            for label, values in sorted(labels.items())
        },
        "by_relation": {
            relation: {"correct": sum(values), "total": len(values), "accuracy": sum(values) / len(values)}
            for relation, values in sorted(groups.items())
        },
        "prediction_counts": dict(Counter(str(row["prediction"]) for row in rows)),
    }


def main() -> int:
    args = parse_args()
    if args.bridge == "on" and args.checkpoint is None:
        raise ValueError("bridge=on requires --checkpoint")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.run_dir / "status.json"
    results_path = args.run_dir / "results.jsonl"
    checkpoint_payload = None
    kwargs = {"drsa_enabled": False, "generation_adapter_enabled": False}
    if args.checkpoint is not None:
        checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        kwargs = checkpoint_model_kwargs(checkpoint_payload)
    dataset = VsrRelationDataset(
        args.data_root / "vsr/annotations/zeroshot" / f"{args.split}.jsonl",
        args.data_root / "vsr/vsr_images.zip",
        image_transform=partial(image_transform, resolution=432),
    )
    total = len(dataset) if args.limit is None else min(len(dataset), args.limit)
    existing = []
    if results_path.is_file():
        existing = [json.loads(line) for line in results_path.read_text().splitlines() if line]
        if len(existing) > total:
            raise RuntimeError("VSR result has more rows than requested")
    status = {
        "status": "initializing", "phase": "loading_model", "split": args.split,
        "bridge": args.bridge, "completed": len(existing), "total": total,
        "percent": 100.0 * len(existing) / total,
    }
    write_json(status_path, status)
    if len(existing) == total:
        summary = {"status": "completed", "split": args.split, "bridge": args.bridge, **aggregate(existing)}
        write_json(args.run_dir / "summary.json", summary)
        status.update(status="completed", phase="completed", percent=100.0, summary=summary)
        write_json(status_path, status)
        return 0
    device, dtype = torch.device(args.device), torch.bfloat16
    model = Showo2Qwen2_5.from_pretrained(
        args.model, use_safetensors=False, local_files_only=True,
        low_cpu_mem_usage=False, torch_dtype=dtype, **kwargs,
    ).to(device=device, dtype=dtype).eval().requires_grad_(False)
    adaptation = None
    bridge_mode = "off"
    if checkpoint_payload is not None:
        adaptation = configure_from_checkpoint(model, checkpoint_payload)
        adaptation.requires_grad_(False)
        adaptation.train(False)
        adaptation.set_enabled(True)
        if args.bridge == "on":
            bridge_mode = (
                "generation_adapter"
                if adaptation.route == "g_adapter"
                else "lora_backbone"
            )
    vae = WanVAE(vae_pth=str(args.vae_path), dtype=dtype, device=device)
    llm_path = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer, token_ids = get_text_tokenizer(
        llm_path, add_showo_tokens=True, return_showo_token_ids=True,
        llm_name=path_to_llm_name[llm_path],
    )
    yes_ids = tokenizer(" yes", add_special_tokens=False)["input_ids"]
    no_ids = tokenizer(" no", add_special_tokens=False)["input_ids"]
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise RuntimeError("VSR yes/no labels must be one token")
    sequence_length = args.max_text_tokens + 734
    rows = list(existing)
    started = time.time()
    try:
        with results_path.open("a", encoding="utf-8") as output:
            for index in range(len(existing), total):
                sample = dataset[index]
                image = sample.pop("image").unsqueeze(0).to(device)
                with torch.no_grad():
                    latents = vae.sample(image.to(dtype).unsqueeze(2), deterministic=True).squeeze(2).to(dtype)
                    sequence = build_understanding_sequence(
                        sample, tokenizer, token_ids, args.max_text_tokens, sequence_length
                    )
                    text_tokens, text_labels, positions, text_mask, image_mask = [
                        value.unsqueeze(0).to(device) for value in sequence
                    ]
                    answer_start = int((text_labels[0] != -100).nonzero()[0])
                    attention_mask = omni_attn_mask_naive(1, sequence_length, positions, device).to(dtype)
                    logits = model.forward_und_only(
                        text_tokens=text_tokens, image_latents=latents,
                        t=torch.ones((1,), device=device, dtype=dtype),
                        attention_mask=attention_mask, text_masks=text_mask,
                        image_masks=image_mask, modality_positions=positions,
                        output_hidden_states=True, max_seq_len=sequence_length,
                        device=device,
                        use_generation_adapter_for_understanding=(
                            bridge_mode == "generation_adapter"
                        ),
                    )
                scores = logits[0, answer_start - 1, [yes_ids[0], no_ids[0]]].float()
                prediction = 1 if int(scores.argmax()) == 0 else 0
                row = {
                    "index": index, "relation": str(sample["relation"]),
                    "label": int(sample["label"]), "prediction": prediction,
                    "correct": prediction == int(sample["label"]),
                    "yes_logit": float(scores[0]), "no_logit": float(scores[1]),
                }
                rows.append(row)
                output.write(json.dumps(row, sort_keys=True) + "\n")
                output.flush()
                elapsed = time.time() - started
                done = index + 1
                status.update(
                    status="running", phase="vsr", completed=done,
                    percent=100.0 * done / total,
                    eta_seconds=elapsed / max(1, done - len(existing)) * (total - done),
                    running_accuracy=sum(bool(value["correct"]) for value in rows) / len(rows),
                )
                write_json(status_path, status)
        summary = {
            "status": "completed", "split": args.split, "bridge": args.bridge,
            "bridge_mode": bridge_mode,
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "checkpoint_step": int(checkpoint_payload.get("step", 0)) if checkpoint_payload else 0,
            **aggregate(rows),
        }
        write_json(args.run_dir / "summary.json", summary)
        status.update(status="completed", phase="completed", percent=100.0, summary=summary)
        write_json(status_path, status)
        return 0
    except BaseException as error:
        status.update(status="failed", phase="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
