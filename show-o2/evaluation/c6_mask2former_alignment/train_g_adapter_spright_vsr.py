#!/usr/bin/env python3
"""One full generation-adapter warmup epoch over SPRIGHT-COCO and VSR."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.drsa_dataset import VsrRelationDataset, iter_spright_shard
from datasets.utils import image_transform
from evaluation.c4_drsa_adapter.train_drsa import (
    build_generation_sequence,
    build_understanding_sequence,
)
from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
from models.generation_adaptation import configure_generation_adaptation
from models.misc import get_text_tokenizer
from transport import create_transport
from utils import path_to_llm_name


SPRIGHT_FULL_COUNT = 40_504
VSR_ZERO_SHOT_TRAIN_COUNT = 3_489
SPRIGHT_SHARD_COUNTS = (4_000,) * 10 + (504,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/data/drsa"))
    parser.add_argument("--model", default="showlab/show-o2-1.5B")
    parser.add_argument("--vae-path", type=Path, default=ROOT / "Wan2.1_VAE.pth")
    parser.add_argument("--spright-limit", type=int, default=SPRIGHT_FULL_COUNT)
    parser.add_argument("--vsr-limit", type=int, default=VSR_ZERO_SHOT_TRAIN_COUNT)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--vsr-loss-weight", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument("--early-save-until", type=int, default=1_000)
    parser.add_argument("--early-save-every", type=int, default=100)
    parser.add_argument("--late-save-every", type=int, default=2_000)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def trim_metrics_for_resume(path: Path, step: int) -> None:
    if not path.is_file():
        return
    kept = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if int(record.get("step", -1)) <= step:
                kept.append(line)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(kept))
    temporary.replace(path)


def task_at_step(step: int, spright_count: int, vsr_count: int) -> str:
    total = spright_count + vsr_count
    if not 1 <= step <= total:
        raise ValueError(f"step {step} is outside [1, {total}]")
    previous_vsr = ((step - 1) * vsr_count) // total
    current_vsr = (step * vsr_count) // total
    return "vsr" if current_vsr > previous_vsr else "spright"


def expected_counts(step: int, spright_count: int, vsr_count: int) -> dict[str, int]:
    total = spright_count + vsr_count
    completed_vsr = (step * vsr_count) // total
    return {"spright": step - completed_vsr, "vsr": completed_vsr}


def iter_spright_from_offset(
    shard_root: Path,
    offset: int,
    transform,
) -> Iterator[dict[str, Any]]:
    if offset < 0 or offset > SPRIGHT_FULL_COUNT:
        raise ValueError(f"invalid SPRIGHT offset: {offset}")
    shards = [shard_root / f"{index}.tar" for index in range(len(SPRIGHT_SHARD_COUNTS))]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing SPRIGHT shards: {missing}")
    remaining_offset = offset
    for shard, expected in zip(shards, SPRIGHT_SHARD_COUNTS):
        if remaining_offset >= expected:
            remaining_offset -= expected
            continue
        local_index = 0
        emitted = 0
        for record in iter_spright_shard(shard):
            if local_index < remaining_offset:
                local_index += 1
                continue
            record["image"] = transform(record["image"])
            emitted += 1
            yield record
        observed = local_index + emitted
        if observed != expected:
            raise RuntimeError(
                f"SPRIGHT shard {shard.name} has {observed} samples; expected {expected}"
            )
        remaining_offset = 0


@torch.no_grad()
def encode_images(vae: WanVAE, images: torch.Tensor, dtype: torch.dtype,
                  device: torch.device) -> torch.Tensor:
    videos = images.to(device=device, dtype=dtype).unsqueeze(2)
    return vae.sample(videos, deterministic=True).squeeze(2).to(dtype)


def grad_norm(parameters) -> float:
    total = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        total = value if total is None else total + value
    return 0.0 if total is None else float(total.sqrt().cpu())


def should_save(step: int, total_steps: int, args: argparse.Namespace) -> bool:
    if step == total_steps or step == args.stop_after_step:
        return True
    if step <= args.early_save_until:
        return step % args.early_save_every == 0
    return step % args.late_save_every == 0


def save_checkpoint(
    adaptation,
    optimizer: AdamW,
    scheduler: LambdaLR,
    step: int,
    counts: Mapping[str, int],
    run_dir: Path,
    metrics: Mapping[str, Any],
    args: argparse.Namespace,
) -> Path:
    checkpoint = run_dir / "checkpoints" / f"step-{step:06d}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    torch.save({
        "format_version": 1,
        "protocol": "showo2-g-adapter-spright-vsr-one-epoch-v1",
        "step": step,
        "counts": dict(counts),
        "adaptation_config": adaptation.config_dict(),
        "adaptation_state": adaptation.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": dict(metrics),
        "data_config": {
            "spright_limit": args.spright_limit,
            "vsr_limit": args.vsr_limit,
            "seed": args.seed,
        },
        "rng_state": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(torch.device(args.device)),
        },
    }, temporary)
    temporary.replace(checkpoint)
    (run_dir / "latest_checkpoint.txt").write_text(str(checkpoint) + "\n")
    return checkpoint


def main() -> int:
    args = parse_args()
    total_steps = args.spright_limit + args.vsr_limit
    if not 1 <= args.spright_limit <= SPRIGHT_FULL_COUNT:
        raise ValueError(f"spright-limit must be in [1, {SPRIGHT_FULL_COUNT}]")
    if not 1 <= args.vsr_limit <= VSR_ZERO_SHOT_TRAIN_COUNT:
        raise ValueError(f"vsr-limit must be in [1, {VSR_ZERO_SHOT_TRAIN_COUNT}]")
    if args.early_save_every < 1 or args.late_save_every < 1:
        raise ValueError("checkpoint intervals must be positive")
    if args.stop_after_step is not None and not 1 <= args.stop_after_step <= total_steps:
        raise ValueError("stop-after-step must be within the configured epoch")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("min-lr-ratio must be within [0, 1]")
    if args.vsr_loss_weight <= 0.0:
        raise ValueError("vsr-loss-weight must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("G-adapter full warmup requires CUDA")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.run_dir / "status.json"
    metrics_path = args.run_dir / "metrics.jsonl"
    started_wall = time.time()
    status: dict[str, Any] = {
        "status": "initializing",
        "phase": "loading_model",
        "step": 0,
        "total_steps": total_steps,
        "percent": 0.0,
        "counts": {"spright": 0, "vsr": 0},
        "dataset_sizes": {
            "spright": args.spright_limit,
            "vsr": args.vsr_limit,
        },
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "run_dir": str(args.run_dir.resolve()),
    }
    atomic_json(status_path, status)

    try:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
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
            drsa_enabled=False,
            generation_adapter_enabled=True,
            output_loading_info=True,
        )
        model.to(device=device, dtype=dtype).requires_grad_(False)
        adaptation = configure_generation_adaptation(model, "g_adapter")
        model.requires_grad_(False)
        adaptation.requires_grad_(True)
        model.eval()
        adaptation.train()
        trainable = [(name, parameter) for name, parameter in model.named_parameters()
                     if parameter.requires_grad]
        adaptation_ids = {id(parameter) for parameter in adaptation.parameters()}
        if not trainable or {id(parameter) for _, parameter in trainable} != adaptation_ids:
            raise RuntimeError("trainable parameters are not exactly the single G-adapter")
        if adaptation.parameter_count != 9_533_428:
            raise RuntimeError(
                f"unexpected G-adapter parameter count: {adaptation.parameter_count}"
            )

        vae = WanVAE(vae_pth=str(args.vae_path), dtype=dtype, device=device)
        optimizer = AdamW(
            adaptation.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
        warmup_steps = max(1, min(args.warmup_steps, total_steps))

        def lr_lambda(index: int) -> float:
            if index < warmup_steps:
                return float(index + 1) / float(warmup_steps)
            progress = (index - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda)
        transport = create_transport(
            path_type="Linear",
            prediction="velocity",
            snr_type="lognorm",
            do_shift=True,
            seq_len=729,
        )

        start_step = 0
        counts = {"spright": 0, "vsr": 0}
        elapsed_offset = 0.0
        if args.resume_checkpoint is not None:
            resume_path = args.resume_checkpoint.expanduser().resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(resume_path)
            payload = torch.load(resume_path, map_location="cpu", weights_only=False)
            if payload.get("protocol") != "showo2-g-adapter-spright-vsr-one-epoch-v1":
                raise ValueError("resume checkpoint has the wrong protocol")
            data_config = payload.get("data_config", {})
            expected_config = {
                "spright_limit": args.spright_limit,
                "vsr_limit": args.vsr_limit,
                "seed": args.seed,
            }
            if data_config != expected_config:
                raise ValueError(
                    f"resume data config {data_config} does not match {expected_config}"
                )
            adaptation.load_state_dict(payload["adaptation_state"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            start_step = int(payload["step"])
            counts = {key: int(value) for key, value in payload["counts"].items()}
            if counts != expected_counts(
                start_step, args.spright_limit, args.vsr_limit
            ):
                raise ValueError("resume checkpoint data counts do not match its step")
            if not 0 < start_step < total_steps:
                raise ValueError("resume step must be inside the configured epoch")
            elapsed_offset = float(payload.get("metrics", {}).get("elapsed_seconds", 0.0))
            rng_state = payload.get("rng_state")
            if not isinstance(rng_state, Mapping):
                raise ValueError("resume checkpoint has no complete RNG state")
            random.setstate(rng_state["python"])
            torch.set_rng_state(rng_state["torch_cpu"])
            torch.cuda.set_rng_state(rng_state["torch_cuda"], device=device)
            trim_metrics_for_resume(metrics_path, start_step)
        else:
            output = adaptation.g_adapter.spatial.output
            if torch.count_nonzero(output.weight).item() or torch.count_nonzero(output.bias).item():
                raise RuntimeError("fresh G-adapter output projection is not zero initialized")
            if metrics_path.exists():
                raise FileExistsError(
                    f"fresh run refuses to append to existing metrics: {metrics_path}"
                )

        transform = partial(image_transform, resolution=432)
        spright_iterator = iter_spright_from_offset(
            args.data_root / "spright_coco" / "data",
            counts["spright"],
            transform,
        )
        vsr = VsrRelationDataset(
            args.data_root / "vsr" / "annotations" / "zeroshot" / "train.jsonl",
            args.data_root / "vsr" / "vsr_images.zip",
            image_transform=transform,
        )
        if len(vsr) != VSR_ZERO_SHOT_TRAIN_COUNT:
            raise RuntimeError(
                f"VSR zero-shot train has {len(vsr)} rows; expected {VSR_ZERO_SHOT_TRAIN_COUNT}"
            )
        if args.vsr_limit > len(vsr):
            raise ValueError("vsr-limit exceeds the local VSR zero-shot train split")

        missing_adapter_keys = [
            key for key in loading_info.get("missing_keys", [])
            if key.startswith("generation_adapter.")
        ]
        status.update({
            "status": "running",
            "phase": "warmup",
            "step": start_step,
            "percent": 100.0 * start_step / total_steps,
            "counts": counts,
            "resume_checkpoint": (
                str(args.resume_checkpoint.expanduser().resolve())
                if args.resume_checkpoint is not None else None
            ),
            "initialization": (
                "resume" if args.resume_checkpoint is not None
                else "original_showo2_zero_output_g_adapter"
            ),
            "trainable_module": "generation_adapter",
            "trainable_parameters": adaptation.parameter_count,
            "base_model_frozen": True,
            "vsr_understanding_bridge": "training_only_explicit_opt_in",
            "missing_generation_adapter_keys": len(missing_adapter_keys),
            "learning_rate": args.learning_rate,
            "vsr_loss_weight": args.vsr_loss_weight,
            "updated_at": now_iso(),
        })
        atomic_json(status_path, status)

        last_metrics: dict[str, Any] = dict(payload.get("metrics", {})) if args.resume_checkpoint else {}
        verified_gradient_tasks = {
            task for task in ("spright", "vsr") if counts[task] > 0
        }
        for step in range(start_step + 1, total_steps + 1):
            task = task_at_step(step, args.spright_limit, args.vsr_limit)
            optimizer.zero_grad(set_to_none=True)
            if task == "spright":
                try:
                    sample = next(spright_iterator)
                except StopIteration as error:
                    raise RuntimeError(
                        f"SPRIGHT exhausted after {counts['spright']} samples"
                    ) from error
                images = sample["image"].unsqueeze(0)
                clean_latents = encode_images(vae, images, dtype, device)
                with torch.no_grad():
                    t, x0, x1 = transport.sample(clean_latents)
                    t, noisy_latents, velocity_target = transport.path_sampler.plan(t, x0, x1)
                    noisy_latents = noisy_latents.to(dtype)
                    velocity_target = velocity_target.to(dtype)
                caption = str(sample["spatial_caption"])
                sequence = build_generation_sequence(
                    caption, tokenizer, token_ids, args.max_text_tokens, sequence_length
                )
                text_tokens, _, positions, text_mask, image_mask = [
                    value.unsqueeze(0).to(device) for value in sequence
                ]
                attention_mask = omni_attn_mask_naive(
                    1, sequence_length, positions, device
                ).to(dtype)
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
                )
                loss = loss_flow.float()
                counts["spright"] += 1
                task_metrics = {
                    "loss_flow": float(loss_flow.detach()),
                    "sample_id": str(sample["sample_id"]),
                    "shard": str(sample["shard"]),
                    "caption": caption,
                }
            else:
                sample = vsr[counts["vsr"]]
                images = sample["image"].unsqueeze(0)
                clean_latents = encode_images(vae, images, dtype, device)
                sequence = build_understanding_sequence(
                    sample, tokenizer, token_ids, args.max_text_tokens, sequence_length
                )
                text_tokens, text_labels, positions, text_mask, image_mask = [
                    value.unsqueeze(0).to(device) for value in sequence
                ]
                attention_mask = omni_attn_mask_naive(
                    1, sequence_length, positions, device
                ).to(dtype)
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
                    use_generation_adapter_for_understanding=True,
                )
                loss = args.vsr_loss_weight * loss_ntp.float()
                counts["vsr"] += 1
                task_metrics = {
                    "loss_ntp": float(loss_ntp.detach()),
                    "loss_ntp_weighted": float(loss.detach()),
                    "vsr_index": counts["vsr"] - 1,
                    "label": int(sample["label"]),
                    "relation": str(sample["relation"]),
                    "caption": str(sample["caption"]),
                }

            if counts != expected_counts(step, args.spright_limit, args.vsr_limit):
                raise RuntimeError(f"data schedule drift at step {step}: {counts}")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {task} loss at step {step}")
            loss.backward()
            unclipped_grad = grad_norm(adaptation.parameters())
            if not math.isfinite(unclipped_grad) or unclipped_grad <= 0.0:
                raise FloatingPointError(
                    f"{task} produced invalid G-adapter gradient {unclipped_grad} at step {step}"
                )
            if task not in verified_gradient_tasks:
                frozen_with_grad = [
                    name for name, parameter in model.named_parameters()
                    if id(parameter) not in adaptation_ids and parameter.grad is not None
                ]
                if frozen_with_grad:
                    raise RuntimeError(
                        f"{task} populated frozen gradients: {frozen_with_grad[:5]}"
                    )
                verified_gradient_tasks.add(task)
            global_grad = float(torch.nn.utils.clip_grad_norm_(
                tuple(adaptation.parameters()), args.grad_clip
            ).detach().cpu())
            optimizer.step()
            scheduler.step()

            elapsed_run = time.time() - started_wall
            elapsed = elapsed_offset + elapsed_run
            completed_this_process = step - start_step
            rate = completed_this_process / max(elapsed_run, 1e-6)
            eta_seconds = (total_steps - step) / max(rate, 1e-12)
            last_metrics = {
                "step": step,
                "task": task,
                "loss": float(loss.detach()),
                "grad_generation_adapter": unclipped_grad,
                "global_grad_norm": global_grad,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "counts": dict(counts),
                "percent": 100.0 * step / total_steps,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta_seconds,
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                **task_metrics,
            }
            append_jsonl(metrics_path, last_metrics)
            checkpoint = None
            if should_save(step, total_steps, args):
                checkpoint = save_checkpoint(
                    adaptation, optimizer, scheduler, step, counts,
                    args.run_dir, last_metrics, args,
                )
            status.update({
                "status": "running",
                "phase": task,
                "step": step,
                "percent": 100.0 * step / total_steps,
                "counts": counts,
                "metrics": last_metrics,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta_seconds,
                "latest_checkpoint": str(checkpoint) if checkpoint else status.get("latest_checkpoint"),
                "verified_gradient_tasks": sorted(verified_gradient_tasks),
                "updated_at": now_iso(),
            })
            atomic_json(status_path, status)
            print(
                f"step={step}/{total_steps} percent={status['percent']:.4f}% "
                f"task={task} loss={last_metrics['loss']:.6f} "
                f"g_grad={unclipped_grad:.4e} counts={counts} "
                f"eta={eta_seconds / 3600:.2f}h",
                flush=True,
            )

            if args.stop_after_step == step and step < total_steps:
                status.update({
                    "status": "stopped",
                    "phase": "intentional_stop",
                    "updated_at": now_iso(),
                })
                atomic_json(status_path, status)
                return 0

        if counts != {"spright": args.spright_limit, "vsr": args.vsr_limit}:
            raise RuntimeError(f"incomplete epoch counts: {counts}")
        if verified_gradient_tasks != {"spright", "vsr"}:
            raise RuntimeError(
                f"did not verify both objective gradients: {verified_gradient_tasks}"
            )
        if args.spright_limit == SPRIGHT_FULL_COUNT:
            try:
                extra = next(spright_iterator)
            except StopIteration:
                extra = None
            if extra is not None:
                raise RuntimeError("SPRIGHT contains more than 40,504 samples")

        status.update({
            "status": "completed",
            "phase": "completed",
            "step": total_steps,
            "percent": 100.0,
            "counts": counts,
            "elapsed_seconds": elapsed_offset + time.time() - started_wall,
            "eta_seconds": 0.0,
            "updated_at": now_iso(),
        })
        atomic_json(status_path, status)
        atomic_json(args.run_dir / "metadata.json", {
            "status": "completed",
            "protocol": "showo2-g-adapter-spright-vsr-one-epoch-v1",
            "initialization": "original Show-o2 plus zero-output G-adapter",
            "adaptation": adaptation.config_dict(),
            "datasets": {
                "spright_coco": {
                    "split": "all local shards 0.tar through 10.tar",
                    "samples": args.spright_limit,
                    "objective": "spatial_caption native velocity loss",
                },
                "vsr": {
                    "split": "zero-shot train",
                    "samples": args.vsr_limit,
                    "objective": "yes/no next-token loss through frozen U",
                    "loss_weight": args.vsr_loss_weight,
                },
            },
            "counts": counts,
            "total_optimizer_steps": total_steps,
            "base_showo2_frozen": True,
            "vae_frozen": True,
            "understanding_head_frozen": True,
            "understanding_bridge_default_inference_behavior_changed": False,
            "checkpoint_policy": {
                "through_step_1000": "every 100 steps",
                "after_step_1000": "every 2000 steps",
                "final": total_steps,
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
