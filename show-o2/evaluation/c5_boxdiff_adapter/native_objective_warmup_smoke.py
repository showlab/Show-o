#!/usr/bin/env python3
"""Warm up the frozen-Show-o2 DRSA adapter on native objectives.

SPRIGHT original captions provide the T2I velocity objective and VSR provides
the MMU yes/no NTP objective. This deliberately excludes BoxDiff losses so
the first run measures only native-objective compatibility.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.drsa_dataset import SprightCocoIterableDataset, VsrRelationDataset
from datasets.utils import image_transform
from evaluation.c4_drsa_adapter.train_drsa import (
    adapter_grad_norm,
    atomic_json,
    build_generation_sequence,
    build_understanding_sequence,
    encode_images,
    now_iso,
    relation_code,
    save_checkpoint,
)
from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
from evaluation.c5_boxdiff_adapter.staged_adapter_loss import (
    StagedAdapterLossConfig,
    compose_staged_loss,
)
from models.misc import get_text_tokenizer
from transport import create_transport
from utils import path_to_llm_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/data/drsa"))
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--model", default="showlab/show-o2-1.5B")
    parser.add_argument("--vae-path", type=Path, default=ROOT / "Wan2.1_VAE.pth")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument("--lr-u", type=float, default=1e-5)
    parser.add_argument("--lr-g", type=float, default=5e-6)
    parser.add_argument("--lr-relation", type=float, default=5e-6)
    parser.add_argument("--t2i-loss-weight", type=float, default=1.0)
    parser.add_argument("--mmu-loss-weight", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--native-steps", type=int, default=500)
    parser.add_argument("--spatial-ramp-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def snapshot_optimizer_groups(optimizer: AdamW) -> dict[str, list[torch.Tensor]]:
    """Capture the exact parameter values that the next AdamW step will mutate."""
    return {
        str(group["name"]): [parameter.detach().clone() for parameter in group["params"]]
        for group in optimizer.param_groups
    }


@torch.no_grad()
def optimizer_group_update_stats(
    optimizer: AdamW,
    snapshots: dict[str, list[torch.Tensor]],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for group in optimizer.param_groups:
        name = str(group["name"])
        before = snapshots[name]
        parameters = list(group["params"])
        if len(before) != len(parameters):
            raise RuntimeError(f"optimizer group {name!r} changed during a step")
        update_sq = torch.zeros((), device=parameters[0].device, dtype=torch.float32)
        parameter_sq = torch.zeros_like(update_sq)
        for parameter, previous in zip(parameters, before):
            current = parameter.detach()
            update_sq.add_((current.float() - previous.float()).square().sum())
            parameter_sq.add_(current.float().square().sum())
        update_norm = update_sq.sqrt()
        parameter_norm = parameter_sq.sqrt()
        output[name] = {
            "update_norm": float(update_norm.cpu()),
            "parameter_norm": float(parameter_norm.cpu()),
            "relative_update": float(
                (update_norm / parameter_norm.clamp_min(1e-12)).cpu()
            ),
        }
    return output


def main() -> int:
    args = parse_args()
    if args.max_steps < 1 or args.save_every < 1:
        raise ValueError("max-steps and save-every must be positive")
    if args.max_steps > args.native_steps:
        raise ValueError(
            "native warm-up wires Phase A only; max-steps must not exceed native-steps"
        )
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("native-objective warm-up requires CUDA")
    if not args.data_manifest.is_file():
        raise FileNotFoundError(args.data_manifest)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.run_dir / "status.json"
    started = time.time()
    status: dict[str, Any] = {
        "status": "initializing",
        "phase": "loading_model",
        "task_mode": "native_objective_warmup",
        "step": 0,
        "total_steps": args.max_steps,
        "percent": 0.0,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "run_dir": str(args.run_dir),
        "data_manifest": str(args.data_manifest),
    }
    atomic_json(status_path, status)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    dtype = torch.bfloat16

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
    if model.drsa_adapter is None:
        raise RuntimeError("drsa_enabled=True did not construct an adapter")
    init_step = 0
    if args.init_checkpoint is not None:
        checkpoint_path = args.init_checkpoint.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("drsa_adapter")
        if not isinstance(state, dict):
            raise ValueError(f"checkpoint has no drsa_adapter state: {checkpoint_path}")
        if any(key.startswith("generation_spatial.") for key in state):
            raise ValueError(
                "native warm-up cannot continue from a legacy generation_spatial checkpoint; "
                "new runs use one generation adapter"
            )
        model.drsa_adapter.load_state_dict(state, strict=True)
        init_step = int(payload.get("step", 0))
        del payload
    model.eval()
    model.drsa_adapter.requires_grad_(True)
    model.drsa_adapter.train()
    vae = WanVAE(vae_pth=str(args.vae_path), dtype=dtype, device=device)

    optimizer = AdamW([
        {"params": model.drsa_adapter.understanding.parameters(), "lr": args.lr_u, "name": "u"},
        {"params": model.drsa_adapter.generation.parameters(), "lr": args.lr_g, "name": "g"},
        {"params": model.drsa_adapter.relation_encoder.parameters(), "lr": args.lr_relation, "name": "relation"},
    ], weight_decay=args.weight_decay)
    warmup = max(1, min(args.warmup_steps, args.max_steps))
    loss_schedule = StagedAdapterLossConfig(
        native_steps=args.native_steps,
        ramp_steps=args.spatial_ramp_steps,
        native_flow_weight=args.t2i_loss_weight,
        mmu_weight=args.mmu_loss_weight,
    )

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)
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
        shuffle_shards=False,
        shuffle_buffer=1,
        seed=args.seed,
    )
    vsr = VsrRelationDataset(
        args.data_root / "vsr" / "annotations" / "zeroshot" / "train.jsonl",
        args.data_root / "vsr" / "vsr_images.zip",
        image_transform=transform,
    )
    spright_loader = DataLoader(
        spright, batch_size=1, num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    vsr_loader = DataLoader(
        vsr, batch_size=1, shuffle=False, num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    spright_iterator = iter(spright_loader)
    vsr_iterator = iter(vsr_loader)
    counts = {"spright_t2i": 0, "vsr_mmu": 0}
    records: list[dict[str, Any]] = []
    status.update({
        "status": "running",
        "phase": "native_objective_warmup",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "base_model_frozen": all(not p.requires_grad for name, p in model.named_parameters() if not name.startswith("drsa_adapter.")),
        "learning_rates": {"u": args.lr_u, "g": args.lr_g, "relation": args.lr_relation},
        "counts": counts,
        "init_checkpoint": (
            str(args.init_checkpoint.expanduser().resolve())
            if args.init_checkpoint is not None else None
        ),
        "init_checkpoint_step": init_step,
        "missing_adapter_keys": sum(
            key.startswith("drsa_adapter.") for key in loading_info.get("missing_keys", [])
        ),
        "updated_at": now_iso(),
    })
    atomic_json(status_path, status)

    try:
        for step in range(1, args.max_steps + 1):
            task = "spright_t2i" if step % 2 == 1 else "vsr_mmu"
            stage_weights = loss_schedule.weights_at(step)
            optimizer.zero_grad(set_to_none=True)
            if task == "spright_t2i":
                try:
                    batch = next(spright_iterator)
                except StopIteration:
                    spright_iterator = iter(spright_loader)
                    batch = next(spright_iterator)
                images = batch["image"]
                caption = str(batch["original_caption"][0])
                clean_latents = encode_images(vae, images, dtype)
                with torch.no_grad():
                    t, x0, x1 = transport.sample(clean_latents)
                    t, noisy_latents, velocity_target = transport.path_sampler.plan(t, x0, x1)
                    noisy_latents = noisy_latents.to(dtype)
                    velocity_target = velocity_target.to(dtype)
                sequence = build_generation_sequence(
                    caption, tokenizer, token_ids, args.max_text_tokens, sequence_length
                )
                text_tokens, _, positions, text_mask, image_mask = [
                    value.unsqueeze(0).to(device) for value in sequence
                ]
                attention_mask = omni_attn_mask_naive(
                    1, sequence_length, positions, device
                ).to(dtype)
                code = relation_code(model, tokenizer, [caption], device)
                _, loss_flow = model(
                    text_tokens=text_tokens,
                    image_latents=noisy_latents,
                    t=t.to(device=device, dtype=dtype),
                    attention_mask=attention_mask,
                    text_masks=text_mask,
                    image_masks=image_mask,
                    image_labels=velocity_target,
                    modality_positions=positions,
                    output_hidden_states=True,
                    max_seq_len=sequence_length,
                    device=device,
                    drsa_relation_codes=code,
                )
                step_losses = {"native_flow": loss_flow.float()}
                loss, weighted_components = compose_staged_loss(
                    step_losses, stage_weights
                )
                counts[task] += 1
                task_metrics = {
                    "loss_flow": float(loss_flow.detach()),
                    "task_loss_weight": args.t2i_loss_weight,
                    "caption": caption,
                }
            else:
                try:
                    batch = next(vsr_iterator)
                except StopIteration:
                    vsr_iterator = iter(vsr_loader)
                    batch = next(vsr_iterator)
                images = batch["image"]
                row = {
                    key: batch[key][0]
                    for key in ("caption", "relation", "subj", "obj", "label")
                }
                row["label"] = int(row["label"])
                clean_latents = encode_images(vae, images, dtype)
                sequence = build_understanding_sequence(
                    row, tokenizer, token_ids, args.max_text_tokens, sequence_length
                )
                text_tokens, text_labels, positions, text_mask, image_mask = [
                    value.unsqueeze(0).to(device) for value in sequence
                ]
                attention_mask = omni_attn_mask_naive(
                    1, sequence_length, positions, device
                ).to(dtype)
                code = relation_code(model, tokenizer, [str(row["relation"])], device)
                _, loss_ntp = model.forward_und_only(
                    text_tokens=text_tokens,
                    image_latents=clean_latents,
                    t=torch.ones((1,), device=device, dtype=dtype),
                    attention_mask=attention_mask,
                    text_masks=text_mask,
                    image_masks=image_mask,
                    text_labels=text_labels,
                    modality_positions=positions,
                    output_hidden_states=True,
                    max_seq_len=sequence_length,
                    device=device,
                    drsa_relation_codes=code,
                )
                step_losses = {"mmu": loss_ntp.float()}
                loss, weighted_components = compose_staged_loss(
                    step_losses, stage_weights
                )
                counts[task] += 1
                task_metrics = {
                    "loss_ntp": float(loss_ntp.detach()),
                    "task_loss_weight": args.mmu_loss_weight,
                    "caption": str(row["caption"]),
                }

            task_metrics["stage"] = stage_weights.as_dict()
            task_metrics["weighted_components"] = {
                key: float(value.detach())
                for key, value in weighted_components.items()
            }
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite {task} loss at step {step}: {float(loss)}"
                )
            loss.backward()
            grad_u = adapter_grad_norm(model.drsa_adapter.understanding)
            grad_g = adapter_grad_norm(model.drsa_adapter.generation)
            grad_relation = adapter_grad_norm(model.drsa_adapter.relation_encoder)
            global_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.drsa_adapter.parameters(), args.grad_clip
                ).detach().cpu()
            )
            clipped_grad_u = adapter_grad_norm(model.drsa_adapter.understanding)
            clipped_grad_g = adapter_grad_norm(model.drsa_adapter.generation)
            clipped_grad_relation = adapter_grad_norm(model.drsa_adapter.relation_encoder)
            applied_lrs = {
                str(group["name"]): float(group["lr"])
                for group in optimizer.param_groups
            }
            parameter_snapshots = snapshot_optimizer_groups(optimizer)
            optimizer.step()
            update_stats = optimizer_group_update_stats(
                optimizer, parameter_snapshots
            )
            scheduler.step()
            elapsed = time.time() - started
            record = {
                "step": step,
                "task": task,
                "loss": float(loss.detach()),
                "grad_u": grad_u,
                "grad_g": grad_g,
                "grad_relation": grad_relation,
                "global_grad_norm": global_grad_norm,
                "clipped_grad_u": clipped_grad_u,
                "clipped_grad_g": clipped_grad_g,
                "clipped_grad_relation": clipped_grad_relation,
                "update_u": update_stats["u"]["update_norm"],
                "update_g": update_stats["g"]["update_norm"],
                "update_relation": update_stats["relation"]["update_norm"],
                "relative_update_u": update_stats["u"]["relative_update"],
                "relative_update_g": update_stats["g"]["relative_update"],
                "relative_update_relation": update_stats["relation"]["relative_update"],
                "lr_u": applied_lrs["u"],
                "lr_g": applied_lrs["g"],
                "lr_relation": applied_lrs["relation"],
                "next_lr_u": float(optimizer.param_groups[0]["lr"]),
                "next_lr_g": float(optimizer.param_groups[1]["lr"]),
                "next_lr_relation": float(optimizer.param_groups[2]["lr"]),
                "counts": dict(counts),
                "elapsed_seconds": elapsed,
                **task_metrics,
            }
            records.append(record)
            status.update({
                "status": "running",
                "phase": f"{stage_weights.stage}:{task}",
                "step": step,
                "total_steps": args.max_steps,
                "percent": 100.0 * step / args.max_steps,
                "counts": counts,
                "metrics": record,
                "elapsed_seconds": elapsed,
                "eta_seconds": elapsed / step * (args.max_steps - step),
                "updated_at": now_iso(),
            })
            atomic_json(status_path, status)
            atomic_json(
                args.run_dir / "native_warmup_metrics.json",
                {"records": records},
            )
            print(
                f"step={step}/{args.max_steps} stage={stage_weights.stage} "
                f"task={task} loss={record['loss']:.6f} "
                f"u_grad={grad_u:.4e} g_grad={grad_g:.4e} "
                f"rel_grad={grad_relation:.4e} "
                f"u_update={record['update_u']:.4e} "
                f"g_update={record['update_g']:.4e}",
                flush=True,
            )
            if step % args.save_every == 0 or step == args.max_steps:
                save_checkpoint(model, optimizer, scheduler, step, args.run_dir, record)

        status.update({
            "status": "completed",
            "phase": "completed",
            "percent": 100.0,
            "updated_at": now_iso(),
            "elapsed_seconds": time.time() - started,
        })
        atomic_json(status_path, status)
        atomic_json(args.run_dir / "metadata.json", {
            "status": "completed",
            "protocol": "native-objective-warmup-proxy-v1",
            "official_native_full_data_available": False,
            "data_manifest": str(args.data_manifest),
            "init_checkpoint": (
                str(args.init_checkpoint.expanduser().resolve())
                if args.init_checkpoint is not None else None
            ),
            "init_checkpoint_step": init_step,
            "losses": ["SPRIGHT original_caption velocity flow", "VSR yes/no NTP"],
            "boxdiff_loss_enabled": False,
            "staged_loss_composer": "evaluation.c5_boxdiff_adapter.staged_adapter_loss",
            "loss_schedule": {
                "native_steps": args.native_steps,
                "spatial_ramp_steps": args.spatial_ramp_steps,
                "phase_a": "native_flow + mmu",
                "phase_b": "native anchor + ramped spatial terms",
                "phase_c": "native anchor + stabilized spatial terms",
            },
            "base_model_frozen": True,
            "learning_rates": {
                "u": args.lr_u,
                "g": args.lr_g,
                "relation": args.lr_relation,
            },
            "loss_weights": {
                "t2i": args.t2i_loss_weight,
                "mmu": args.mmu_loss_weight,
            },
        })
        return 0
    except BaseException as error:
        status.update({
            "status": "failed",
            "phase": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "updated_at": now_iso(),
        })
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
