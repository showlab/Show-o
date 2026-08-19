#!/usr/bin/env python3
"""Adapter-only DRSA training on SPRIGHT-COCO and VSR."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from datasets.drsa_dataset import SprightCocoIterableDataset, VsrRelationDataset
from datasets.utils import format_sequence_gen_qwen2_5, image_transform
from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
from models.misc import get_text_tokenizer
from transport import create_transport
from utils import path_to_llm_name


RELATION_PHRASES = (
    "at the left side of",
    "at the right side of",
    "in front of",
    "far away from",
    "on top of",
    "larger than",
    "smaller than",
    "next to",
    "left of",
    "right of",
    "above",
    "below",
    "behind",
    "under",
    "inside",
    "outside",
    "near",
    "far from",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/data/drsa"))
    parser.add_argument("--model", default="showlab/show-o2-1.5B")
    parser.add_argument("--vae-path", type=Path, default=Path("/root/autodl-tmp/Show-o/show-o2/Wan2.1_VAE.pth"))
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument("--retention-every", type=int, default=4)
    parser.add_argument("--retention-weight", type=float, default=0.2)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mode", choices=("smoke", "train"), default="train")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def extract_relation_text(caption: str) -> str:
    lowered = caption.lower()
    matches = [phrase for phrase in RELATION_PHRASES if phrase in lowered]
    return " ; ".join(matches) if matches else caption


def build_generation_sequence(caption, tokenizer, token_ids, max_text_tokens, sequence_length):
    caption_ids = tokenizer(
        caption, add_special_tokens=False, truncation=True, max_length=max_text_tokens
    )["input_ids"]
    return format_sequence_gen_qwen2_5(
        caption_ids,
        ([], [], []),
        token_ids["bos_id"],
        token_ids["eos_id"],
        token_ids["boi_id"],
        token_ids["eoi_id"],
        tokenizer.pad_token_id,
        token_ids["img_pad_id"],
        730,
        sequence_length,
        0,
    )


def build_understanding_sequence(row, tokenizer, token_ids, max_text_tokens, sequence_length):
    prompt = f"Statement: {row['caption']} Answer yes or no:"
    prompt_ids = tokenizer(
        prompt, add_special_tokens=False, truncation=True, max_length=max_text_tokens - 4
    )["input_ids"]
    answer = " yes" if int(row["label"]) == 1 else " no"
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"][:3]
    prefix = [token_ids["bos_id"], token_ids["boi_id"]]
    image = [token_ids["img_pad_id"]] * 730
    tokens = prefix + image + [token_ids["eoi_id"]] + prompt_ids + answer_ids + [token_ids["eos_id"]]
    if len(tokens) > sequence_length:
        raise ValueError(f"understanding sequence is too long: {len(tokens)}")
    labels = [-100] * len(tokens)
    answer_start = len(prefix) + len(image) + 1 + len(prompt_ids)
    labels[answer_start : answer_start + len(answer_ids)] = answer_ids
    labels[answer_start + len(answer_ids)] = token_ids["eos_id"]
    padding = sequence_length - len(tokens)
    tokens.extend([tokenizer.pad_token_id] * padding)
    labels.extend([-100] * padding)
    text_tokens = torch.tensor(tokens, dtype=torch.long)
    text_labels = torch.tensor(labels, dtype=torch.long)
    modality_positions = torch.tensor([[2, 730]], dtype=torch.long)
    text_mask = ((text_tokens != token_ids["img_pad_id"]) & (text_tokens != tokenizer.pad_token_id)).long()
    image_mask = (text_tokens == token_ids["img_pad_id"]).long()
    return text_tokens, text_labels, modality_positions, text_mask, image_mask


def relation_code(model, tokenizer, texts, device):
    encoded = tokenizer(
        list(texts),
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        features = model.showo.model.embed_tokens(token_ids)
    return model.drsa_adapter.encode_relation(features.detach(), mask)


def encode_images(vae, images, dtype):
    with torch.no_grad():
        videos = images.to(device="cuda", dtype=dtype).unsqueeze(2)
        latents = vae.sample(videos, deterministic=True)
        return latents.squeeze(2).to(dtype)


def adapter_grad_norm(module) -> float:
    total = torch.zeros((), device="cuda", dtype=torch.float32)
    for parameter in module.parameters():
        if parameter.grad is not None:
            total = total + parameter.grad.detach().float().square().sum()
    return float(total.sqrt().cpu())


def save_checkpoint(model, optimizer, scheduler, step, run_dir, metrics):
    checkpoint = run_dir / "checkpoints" / f"step-{step:06d}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.drsa_adapter.state_dict().items()}
    torch.save(
        {
            "step": step,
            "drsa_adapter": state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
        },
        checkpoint,
    )
    (run_dir / "latest_checkpoint.txt").write_text(str(checkpoint) + "\n")
    return checkpoint


def main() -> int:
    args = parse_args()
    if args.max_steps < 1:
        raise ValueError("max_steps must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.run_dir / "status.json"
    started = time.time()
    status = {
        "status": "initializing",
        "mode": args.mode,
        "step": 0,
        "total_steps": args.max_steps,
        "percent": 0.0,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "run_dir": str(args.run_dir),
    }
    atomic_json(status_path, status)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    dtype = torch.bfloat16
    device = torch.device("cuda")
    llm_path = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer, token_ids = get_text_tokenizer(
        llm_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[llm_path],
    )
    sequence_length = args.max_text_tokens + 734

    model, loading_info = Showo2Qwen2_5.from_pretrained(
        args.model,
        use_safetensors=False,
        local_files_only=True,
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
        drsa_enabled=True,
        output_loading_info=True,
    )
    model.to(device=device, dtype=dtype)
    model.requires_grad_(False)
    model.eval()
    model.drsa_adapter.requires_grad_(True)
    model.drsa_adapter.train()
    vae = WanVAE(vae_pth=str(args.vae_path), dtype=dtype, device=device)

    optimizer = AdamW(
        model.drsa_adapter.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    warmup = max(1, min(args.warmup_steps, args.max_steps))
    def lr_lambda(step):
        if step < warmup:
            return float(step + 1) / float(warmup)
        remaining = max(1, args.max_steps - warmup)
        return max(0.1, float(args.max_steps - step) / float(remaining))
    scheduler = LambdaLR(optimizer, lr_lambda)
    start_step = 0
    resume_metrics = {}
    if args.resume_checkpoint is not None:
        checkpoint_path = args.resume_checkpoint.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "drsa_adapter" not in payload or "optimizer" not in payload or "scheduler" not in payload:
            raise KeyError(f"Invalid resume checkpoint: {checkpoint_path}")
        model.drsa_adapter.load_state_dict(payload["drsa_adapter"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_step = int(payload.get("step", 0))
        resume_metrics = dict(payload.get("metrics", {}))
        if start_step < 0 or start_step >= args.max_steps:
            raise ValueError(
                f"resume step {start_step} must be in [0, max_steps={args.max_steps})"
            )
        scheduler.last_epoch = start_step
        scheduler._step_count = start_step + 1
        resumed_lr = float(args.learning_rate * lr_lambda(start_step))
        for group in optimizer.param_groups:
            group["lr"] = resumed_lr
        scheduler._last_lr = [resumed_lr for _ in optimizer.param_groups]
    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        snr_type="lognorm",
        do_shift=True,
        seq_len=729,
    )

    transform = partial(image_transform, resolution=432)
    spright = SprightCocoIterableDataset(
        args.data_root / "spright_coco" / "data",
        image_transform=transform,
        shuffle_shards=True,
        shuffle_buffer=64,
        seed=args.seed,
    )
    vsr = VsrRelationDataset(
        args.data_root / "vsr" / "annotations" / "zeroshot" / "train.jsonl",
        args.data_root / "vsr" / "vsr_images.zip",
        image_transform=transform,
    )
    spright_loader = DataLoader(
        spright, batch_size=1, num_workers=args.workers, persistent_workers=args.workers > 0
    )
    vsr_loader = DataLoader(
        vsr, batch_size=1, shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0
    )
    spright_iterator = iter(spright_loader)
    vsr_iterator = iter(vsr_loader)
    initial_outputs = {
        "u": model.drsa_adapter.understanding.output.weight.detach().float().clone(),
        "g": model.drsa_adapter.generation.output.weight.detach().float().clone(),
    }
    counts = {"spright": 0, "vsr": 0}
    last_metrics = {}

    status.update({
        "status": "running",
        "updated_at": now_iso(),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "resume_checkpoint": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None,
        "resume_step": start_step,
        "missing_adapter_keys": sum(
            key.startswith("drsa_adapter.") for key in loading_info.get("missing_keys", [])
        ),
    })
    atomic_json(status_path, status)

    for step in range(start_step + 1, args.max_steps + 1):
        task = "vsr" if step % 3 == 2 else "spright"
        optimizer.zero_grad(set_to_none=True)
        if task == "spright":
            try:
                batch = next(spright_iterator)
            except StopIteration:
                spright_iterator = iter(spright_loader)
                batch = next(spright_iterator)
            images = batch["image"]
            spatial_caption = batch["spatial_caption"][0]
            original_caption = batch["original_caption"][0]
            clean_latents = encode_images(vae, images, dtype)
            with torch.no_grad():
                t, x0, x1 = transport.sample(clean_latents)
                t, noisy_latents, velocity_target = transport.path_sampler.plan(t, x0, x1)
                noisy_latents = noisy_latents.to(dtype)
                velocity_target = velocity_target.to(dtype)
            sequence = build_generation_sequence(
                spatial_caption, tokenizer, token_ids, args.max_text_tokens, sequence_length
            )
            text_tokens, _, modality_positions, text_mask, image_mask = [
                value.unsqueeze(0).to(device) for value in sequence
            ]
            attention_mask = omni_attn_mask_naive(
                1, sequence_length, modality_positions, device
            ).to(dtype)
            relation_text = extract_relation_text(spatial_caption)
            code = relation_code(model, tokenizer, [relation_text], device)
            caption_code = relation_code(model, tokenizer, [spatial_caption], device)
            _, loss_flow = model(
                text_tokens=text_tokens,
                image_latents=noisy_latents,
                t=t.to(device=device, dtype=dtype),
                attention_mask=attention_mask,
                text_masks=text_mask,
                image_masks=image_mask,
                image_labels=velocity_target,
                modality_positions=modality_positions,
                output_hidden_states=True,
                max_seq_len=sequence_length,
                device=device,
                drsa_relation_codes=code,
            )
            loss_alignment = 1.0 - F.cosine_similarity(code.float(), caption_code.float()).mean()
            loss_retention = torch.zeros((), device=device)
            if args.retention_every > 0 and step % args.retention_every == 0:
                retention_sequence = build_generation_sequence(
                    original_caption, tokenizer, token_ids, args.max_text_tokens, sequence_length
                )
                retention_tokens, _, retention_positions, retention_text_mask, retention_image_mask = [
                    value.unsqueeze(0).to(device) for value in retention_sequence
                ]
                retention_mask = omni_attn_mask_naive(
                    1, sequence_length, retention_positions, device
                ).to(dtype)
                _, loss_retention = model(
                    text_tokens=retention_tokens,
                    image_latents=noisy_latents,
                    t=t.to(device=device, dtype=dtype),
                    attention_mask=retention_mask,
                    text_masks=retention_text_mask,
                    image_masks=retention_image_mask,
                    image_labels=velocity_target,
                    modality_positions=retention_positions,
                    output_hidden_states=True,
                    max_seq_len=sequence_length,
                    device=device,
                    drsa_relation_codes=code,
                )
            loss = loss_flow.float() + args.alignment_weight * loss_alignment
            loss = loss + args.retention_weight * loss_retention.float()
            counts["spright"] += 1
            task_metrics = {
                "loss_flow": float(loss_flow.detach()),
                "loss_alignment": float(loss_alignment.detach()),
                "loss_retention": float(loss_retention.detach()),
            }
        else:
            try:
                batch = next(vsr_iterator)
            except StopIteration:
                vsr_iterator = iter(vsr_loader)
                batch = next(vsr_iterator)
            images = batch["image"]
            clean_latents = encode_images(vae, images, dtype)
            row = {key: batch[key][0] if isinstance(batch[key], list) else batch[key][0] for key in ("caption", "relation", "subj", "obj")}
            row["label"] = int(batch["label"][0])
            sequence = build_understanding_sequence(
                row, tokenizer, token_ids, args.max_text_tokens, sequence_length
            )
            text_tokens, text_labels, modality_positions, text_mask, image_mask = [
                value.unsqueeze(0).to(device) for value in sequence
            ]
            attention_mask = omni_attn_mask_naive(
                1, sequence_length, modality_positions, device
            ).to(dtype)
            code = relation_code(model, tokenizer, [str(row["relation"])], device)
            t = torch.ones((1,), device=device, dtype=dtype)
            _, loss_ntp = model.forward_und_only(
                text_tokens=text_tokens,
                image_latents=clean_latents,
                t=t,
                attention_mask=attention_mask,
                text_masks=text_mask,
                image_masks=image_mask,
                text_labels=text_labels,
                modality_positions=modality_positions,
                output_hidden_states=True,
                max_seq_len=sequence_length,
                device=device,
                drsa_relation_codes=code,
            )
            loss = loss_ntp.float()
            counts["vsr"] += 1
            task_metrics = {"loss_ntp": float(loss_ntp.detach())}

        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {float(loss)}")
        loss.backward()
        grad_u = adapter_grad_norm(model.drsa_adapter.understanding)
        grad_g = adapter_grad_norm(model.drsa_adapter.generation)
        grad_relation = adapter_grad_norm(model.drsa_adapter.relation_encoder)
        torch.nn.utils.clip_grad_norm_(model.drsa_adapter.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        elapsed = time.time() - started
        eta = elapsed / step * (args.max_steps - step)
        last_metrics = {
            "loss": float(loss.detach()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_u": grad_u,
            "grad_g": grad_g,
            "grad_relation": grad_relation,
            **task_metrics,
        }
        status.update({
            "status": "running",
            "phase": task,
            "step": step,
            "total_steps": args.max_steps,
            "percent": 100.0 * step / args.max_steps,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "counts": counts,
            "metrics": last_metrics,
            "updated_at": now_iso(),
        })
        atomic_json(status_path, status)
        print(
            f"step={step}/{args.max_steps} task={task} loss={last_metrics['loss']:.6f} "
            f"u_grad={grad_u:.4e} g_grad={grad_g:.4e} rel_grad={grad_relation:.4e}",
            flush=True,
        )
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(model, optimizer, scheduler, step, args.run_dir, last_metrics)

    delta_u = float((
        model.drsa_adapter.understanding.output.weight.detach().float() - initial_outputs["u"]
    ).abs().sum().cpu())
    delta_g = float((
        model.drsa_adapter.generation.output.weight.detach().float() - initial_outputs["g"]
    ).abs().sum().cpu())
    if delta_u <= 0.0 or delta_g <= 0.0:
        raise RuntimeError(f"adapter outputs did not both update: U={delta_u}, G={delta_g}")
    status.update({
        "status": "completed",
        "phase": "completed",
        "percent": 100.0,
        "updated_at": now_iso(),
        "parameter_delta": {"u_output": delta_u, "g_output": delta_g},
    })
    atomic_json(status_path, status)
    return 0


if __name__ == "__main__":
    args_for_error = None
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        try:
            parsed = parse_args()
            parsed.run_dir.mkdir(parents=True, exist_ok=True)
            status_file = parsed.run_dir / "status.json"
            failed_status = {}
            if status_file.is_file():
                try:
                    failed_status = json.loads(status_file.read_text())
                except (OSError, json.JSONDecodeError):
                    failed_status = {}
            failed_status.update({
                "status": "failed",
                "updated_at": now_iso(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            })
            atomic_json(status_file, failed_status)
        finally:
            traceback.print_exc()
        raise
