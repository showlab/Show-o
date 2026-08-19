#!/usr/bin/env python3
"""Train only the DRSA understanding route on VSR.

The Show-o2 backbone and the generation route are frozen. Checkpoints are
saved at a regular interval so a separate sweep process can evaluate each
stage without changing the mixed SPRIGHT/VSR recipe.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from functools import partial
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from datasets.drsa_dataset import VsrRelationDataset
from datasets.utils import image_transform
from evaluation.c4_drsa_adapter.train_drsa import (
    adapter_grad_norm,
    atomic_json,
    build_understanding_sequence,
    encode_images,
    now_iso,
    relation_code,
    save_checkpoint,
)
from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
from models.misc import get_text_tokenizer
from utils import path_to_llm_name


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/data/drsa"))
    parser.add_argument("--model", default="showlab/show-o2-1.5B")
    parser.add_argument("--vae-path", type=Path, default=ROOT / "Wan2.1_VAE.pth")
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_steps < 1 or args.save_every < 1:
        raise ValueError("max_steps and save_every must be positive")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("VSR-only training requires CUDA")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.run_dir / "status.json"
    started = time.time()
    atomic_json(status_path, {
        "status": "initializing", "phase": "loading_model", "task_mode": "vsr_only",
        "step": 0, "total_steps": args.max_steps, "percent": 0.0,
        "started_at": now_iso(), "updated_at": now_iso(), "run_dir": str(args.run_dir),
    })
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    dtype = torch.bfloat16
    llm_path = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer, token_ids = get_text_tokenizer(
        llm_path, add_showo_tokens=True, return_showo_token_ids=True,
        llm_name=path_to_llm_name[llm_path],
    )
    sequence_length = args.max_text_tokens + 734
    model, loading_info = Showo2Qwen2_5.from_pretrained(
        args.model, use_safetensors=False, local_files_only=True,
        low_cpu_mem_usage=False, torch_dtype=dtype, drsa_enabled=True,
        output_loading_info=True,
    )
    model.to(device=device, dtype=dtype)
    model.requires_grad_(False)
    model.drsa_adapter.requires_grad_(True)
    model.drsa_adapter.train()
    model.drsa_adapter.generation.requires_grad_(False)
    vae = WanVAE(vae_pth=str(args.vae_path), dtype=dtype, device=device)
    optimizer = AdamW(
        model.drsa_adapter.parameters(), lr=args.learning_rate,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay,
    )
    warmup = max(1, min(args.warmup_steps, args.max_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        remaining = max(1, args.max_steps - warmup)
        return max(0.1, float(args.max_steps - step) / float(remaining))

    scheduler = LambdaLR(optimizer, lr_lambda)
    dataset = VsrRelationDataset(
        args.data_root / "vsr/annotations/zeroshot/train.jsonl",
        args.data_root / "vsr/vsr_images.zip",
        image_transform=partial(image_transform, resolution=432),
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    iterator = iter(loader)
    status = {
        "status": "running", "phase": "vsr_only", "task_mode": "vsr_only",
        "step": 0, "total_steps": args.max_steps, "percent": 0.0,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "generation_requires_grad": any(p.requires_grad for p in model.drsa_adapter.generation.parameters()),
        "missing_adapter_keys": sum(key.startswith("drsa_adapter.") for key in loading_info.get("missing_keys", [])),
        "counts": {"vsr": 0}, "updated_at": now_iso(),
    }
    atomic_json(status_path, status)
    last_metrics = {}
    try:
        for step in range(1, args.max_steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            images = batch["image"]
            clean_latents = encode_images(vae, images, dtype)
            row = {key: batch[key][0] if isinstance(batch[key], list) else batch[key][0]
                   for key in ("caption", "relation", "subj", "obj")}
            row["label"] = int(batch["label"][0])
            sequence = build_understanding_sequence(
                row, tokenizer, token_ids, args.max_text_tokens, sequence_length,
            )
            text_tokens, text_labels, positions, text_mask, image_mask = [
                value.unsqueeze(0).to(device) for value in sequence
            ]
            attention_mask = omni_attn_mask_naive(
                1, sequence_length, positions, device,
            ).to(dtype)
            code = relation_code(model, tokenizer, [str(row["relation"])], device)
            _, loss_ntp = model.forward_und_only(
                text_tokens=text_tokens, image_latents=clean_latents,
                t=torch.ones((1,), device=device, dtype=dtype),
                attention_mask=attention_mask, text_masks=text_mask,
                image_masks=image_mask, text_labels=text_labels,
                modality_positions=positions, output_hidden_states=True,
                max_seq_len=sequence_length, device=device,
                drsa_relation_codes=code,
            )
            if not torch.isfinite(loss_ntp):
                raise FloatingPointError(f"non-finite VSR loss at step {step}: {float(loss_ntp)}")
            loss_ntp.float().backward()
            grad_u = adapter_grad_norm(model.drsa_adapter.understanding)
            grad_g = adapter_grad_norm(model.drsa_adapter.generation)
            grad_relation = adapter_grad_norm(model.drsa_adapter.relation_encoder)
            torch.nn.utils.clip_grad_norm_(model.drsa_adapter.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            status.update({
                "status": "running", "phase": "vsr_only", "step": step,
                "total_steps": args.max_steps, "percent": 100.0 * step / args.max_steps,
                "counts": {"vsr": step}, "metrics": {
                    "loss_ntp": float(loss_ntp.detach()), "loss": float(loss_ntp.detach()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "grad_u": grad_u, "grad_g": grad_g, "grad_relation": grad_relation,
                }, "elapsed_seconds": time.time() - started,
                "eta_seconds": (time.time() - started) / step * (args.max_steps - step),
                "updated_at": now_iso(),
            })
            atomic_json(status_path, status)
            print(
                f"step={step}/{args.max_steps} task=vsr_only loss={float(loss_ntp):.6f} "
                f"u_grad={grad_u:.4e} g_grad={grad_g:.4e} rel_grad={grad_relation:.4e}",
                flush=True,
            )
            if step % args.save_every == 0 or step == args.max_steps:
                save_checkpoint(model, optimizer, scheduler, step, args.run_dir, status["metrics"])
    except BaseException as error:
        status.update({"status": "failed", "phase": "failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(), "updated_at": now_iso()})
        atomic_json(status_path, status)
        raise
    status.update({"status": "completed", "phase": "completed", "step": args.max_steps, "percent": 100.0, "updated_at": now_iso()})
    atomic_json(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
