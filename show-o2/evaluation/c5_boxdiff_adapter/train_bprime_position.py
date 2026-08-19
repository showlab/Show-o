#!/usr/bin/env python3
"""Train Show-o2's single DRSA generation route on real COCO boxes."""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.drsa_dataset import BprimeCocoBoxDataset
from datasets.utils import image_transform
from evaluation.c4_drsa_adapter.train_drsa import (
    adapter_grad_norm,
    atomic_json,
    encode_images,
    now_iso,
    relation_code,
    save_checkpoint,
)
from evaluation.c5_boxdiff_adapter.boxdiff_adapter_smoke import phrase_embeddings
from evaluation.c5_boxdiff_adapter.bprime_position_loss import (
    LOSS_NAMES,
    calibrate_weights,
    compose_position_loss,
    position_only_losses,
)
from evaluation.c5_boxdiff_adapter.native_objective_warmup_smoke import (
    optimizer_group_update_stats,
    snapshot_optimizer_groups,
)
from evaluation.c5_boxdiff_adapter.staged_spatial_adapter_smoke import (
    build_generation_inputs,
    model_velocity,
)
from guidance.layout import layout_from_mapping, map_layout_tokens
from models import Showo2Qwen2_5, WanVAE
from models.misc import get_text_tokenizer
from transport import create_transport
from utils import path_to_llm_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, default=Path("/root/autodl-tmp/data/drsa/spright_coco/data"))
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--model", default="showlab/show-o2-1.5B")
    parser.add_argument("--vae-path", type=Path, default=ROOT / "Wan2.1_VAE.pth")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--monitor-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument(
        "--lr-generation",
        "--lr-spatial",
        dest="lr_generation",
        type=float,
        default=1e-5,
        help="Learning rate for the single trainable DRSA generation route.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--map-temperature", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_b1_checkpoint(model: Showo2Qwen2_5, path: Path) -> int:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("drsa_adapter")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no drsa_adapter state: {path}")
    if any(key.startswith("generation_spatial.") for key in state):
        raise ValueError("single-G training cannot initialize from a legacy generation_spatial checkpoint")
    model.drsa_adapter.load_state_dict(state, strict=True)
    step = int(payload.get("step", 0))
    del payload
    return step


def load_position_checkpoint(model: Showo2Qwen2_5, path: Path) -> tuple[int, dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("drsa_adapter")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no drsa_adapter state: {path}")
    if any(key.startswith("generation_spatial.") for key in state):
        raise ValueError(f"cannot resume single-G training from a legacy checkpoint: {path}")
    model.drsa_adapter.load_state_dict(state, strict=True)
    step = int(payload.get("step", 0))
    return step, payload


def prepare_layout(row: dict[str, Any], tokenizer: Any):
    payload = {
        "relation": str(row["relation"]),
        "objects": [
            {"phrase": str(item["phrase"]), "box": list(item["box"])}
            for item in row["objects"]
        ],
    }
    return map_layout_tokens(layout_from_mapping(str(row["prompt"]), payload), tokenizer)


def main() -> int:
    args = parse_args()
    if args.max_steps < 1 or args.save_every < 1 or args.monitor_every < 1:
        raise ValueError("max-steps, save-every, and monitor-every must be positive")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("B-prime position training requires CUDA")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.run_dir / "status.json"
    started = time.time()
    status: dict[str, Any] = {
        "status": "initializing",
        "phase": "loading_model",
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
    device, dtype = torch.device(args.device), torch.bfloat16

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
    init_step = load_b1_checkpoint(model, args.init_checkpoint)
    generation = model.drsa_adapter.generation
    teacher_generation = copy.deepcopy(generation).to(device=device, dtype=dtype)
    teacher_generation.requires_grad_(False).eval()
    resume_step, resume_payload = 0, None
    if args.resume_checkpoint is not None:
        resume_step, resume_payload = load_position_checkpoint(model, args.resume_checkpoint)
        if resume_step >= args.max_steps:
            raise ValueError(f"resume step {resume_step} must be below max-steps {args.max_steps}")
        generation = model.drsa_adapter.generation
    model.requires_grad_(False)
    generation.requires_grad_(True)
    model.eval()
    generation.train()
    expected_prefix = "drsa_adapter.generation."
    active_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not active_names or any(not name.startswith(expected_prefix) for name in active_names):
        raise AssertionError(f"unexpected B2 trainable parameters: {active_names[:20]}")

    optimizer = AdamW(
        [{"params": generation.parameters(), "lr": args.lr_generation, "name": "generation"}],
        weight_decay=args.weight_decay,
    )
    warmup = max(1, min(args.warmup_steps, args.max_steps))

    def lr_lambda(step: int) -> float:
        return float(step + 1) / float(warmup) if step < warmup else 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)
    if resume_payload is not None:
        optimizer_state = resume_payload.get("optimizer")
        scheduler_state = resume_payload.get("scheduler")
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
        if isinstance(scheduler_state, dict):
            scheduler.load_state_dict(scheduler_state)
        del resume_payload
    vae = WanVAE(vae_pth=str(args.vae_path), dtype=dtype, device=device)
    transport = create_transport(
        path_type="Linear", prediction="velocity", snr_type="lognorm", do_shift=True, seq_len=729
    )
    dataset = BprimeCocoBoxDataset(
        args.manifest,
        args.shard_root,
        image_transform=partial(image_transform, resolution=432),
    )
    order = list(range(len(dataset)))
    random.Random(args.seed).shuffle(order)
    hook_state: dict[str, torch.Tensor] = {}

    def teacher_pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        image_tokens, relation_codes = inputs[:2]
        with torch.no_grad():
            hook_state["base_hidden"] = teacher_generation(
                image_tokens.detach(),
                relation_codes.detach(),
                grid_height=27,
                grid_width=27,
            )

    def generation_hook(_module: Any, _inputs: Any, output: Any) -> None:
        hook_state["spatial_hidden"] = output[0] if isinstance(output, tuple) else output

    teacher_handle = generation.register_forward_pre_hook(teacher_pre_hook)
    generation_handle = generation.register_forward_hook(generation_hook)

    def forward_row(row: dict[str, Any]):
        layout = prepare_layout(row, tokenizer)
        inputs = build_generation_inputs(
            str(row["prompt"]), tokenizer, token_ids, args.max_text_tokens,
            sequence_length, device, dtype,
        )
        code = relation_code(
            model,
            tokenizer,
            [f"{layout.objects[0].phrase} {layout.relation} {layout.objects[1].phrase}"],
            device,
        )
        phrases = phrase_embeddings(model, tokenizer, layout, device)
        clean_latents = encode_images(vae, row["image"].unsqueeze(0), dtype)
        with torch.no_grad():
            t, x0, x1 = transport.sample(clean_latents)
            t, noisy_latents, velocity_target = transport.path_sampler.plan(t, x0, x1)
            timestep = t.to(device=device, dtype=dtype)
            noisy_latents, velocity_target = noisy_latents.to(dtype), velocity_target.to(dtype)
        hook_state.clear()
        velocity, _ = model_velocity(
            model, inputs, noisy_latents, timestep, code, sequence_length, device
        )
        if "base_hidden" not in hook_state or "spatial_hidden" not in hook_state:
            raise RuntimeError("B2 hooks did not capture both teacher and active generation hidden states")
        losses, details = position_only_losses(
            hook_state["spatial_hidden"],
            hook_state["base_hidden"],
            layout,
            phrases,
            temperature=args.map_temperature,
        )
        return losses, details, velocity, velocity_target, inputs, noisy_latents, timestep, code

    records = []
    calibration_row = dataset[order[0]]
    calibration_losses, _details, *_unused = forward_row(calibration_row)
    loss_weights, calibration_grad_norms = calibrate_weights(calibration_losses, generation)
    optimizer.zero_grad(set_to_none=True)
    del calibration_losses, _details, _unused
    torch.cuda.empty_cache()
    status.update({
        "status": "running",
        "phase": "position_only",
        "trainable_parameters": sum(parameter.numel() for parameter in generation.parameters()),
        "trainable_parameter_names": active_names,
        "base_model_frozen": all(
            not parameter.requires_grad
            for name, parameter in model.named_parameters()
            if not name.startswith(expected_prefix)
        ),
        "init_checkpoint": str(args.init_checkpoint.expanduser().resolve()),
        "init_checkpoint_step": init_step,
        "manifest": str(args.manifest.resolve()),
        "manifest_rows": len(dataset),
        "loss_names": list(LOSS_NAMES),
        "loss_weights": loss_weights,
        "calibration_grad_norms": calibration_grad_norms,
        "lr_generation": args.lr_generation,
        "generation_adapter_policy": "single_generation_route",
        "missing_adapter_keys": sum(
            key.startswith("drsa_adapter.") for key in loading_info.get("missing_keys", [])
        ),
        "updated_at": now_iso(),
    })
    status.update({
        "step": resume_step,
        "resume_checkpoint": str(args.resume_checkpoint.expanduser().resolve()) if args.resume_checkpoint else None,
        "resume_step": resume_step,
        "percent": 100.0 * resume_step / args.max_steps,
        "updated_at": now_iso(),
    })
    atomic_json(status_path, status)

    try:
        for step in range(resume_step + 1, args.max_steps + 1):
            row = dataset[order[(step - 1) % len(order)]]
            optimizer.zero_grad(set_to_none=True)
            (
                raw_losses, details, velocity, velocity_target, inputs,
                noisy_latents, timestep, code,
            ) = forward_row(row)
            loss, weighted = compose_position_loss(raw_losses, loss_weights)
            flow_mse = F.mse_loss(velocity.detach().float(), velocity_target.float())
            teacher_flow_mse, velocity_drift = None, None
            if step == 1 or step % args.monitor_every == 0 or step == args.max_steps:
                active_generation = model.drsa_adapter.generation
                model.drsa_adapter.generation = teacher_generation
                try:
                    with torch.no_grad():
                        teacher_velocity, _ = model_velocity(
                            model, inputs, noisy_latents, timestep, code, sequence_length, device
                        )
                finally:
                    model.drsa_adapter.generation = active_generation
                teacher_flow_mse = float(F.mse_loss(
                    teacher_velocity.float(), velocity_target.float()
                ).cpu())
                velocity_drift = float(F.mse_loss(
                    velocity.detach().float(), teacher_velocity.float()
                ).cpu())
            loss.backward()
            frozen_with_grad = [
                name for name, parameter in model.named_parameters()
                if not parameter.requires_grad and parameter.grad is not None
            ]
            if frozen_with_grad:
                raise AssertionError(f"B2 produced gradients on frozen parameters: {frozen_with_grad[:10]}")
            raw_grad = adapter_grad_norm(generation)
            if raw_grad == 0.0:
                raise AssertionError("B2 position loss produced zero generation gradient")
            global_grad = float(torch.nn.utils.clip_grad_norm_(
                generation.parameters(), args.grad_clip
            ).detach().cpu())
            clipped_grad = adapter_grad_norm(generation)
            applied_lr = float(optimizer.param_groups[0]["lr"])
            snapshots = snapshot_optimizer_groups(optimizer)
            optimizer.step()
            updates = optimizer_group_update_stats(optimizer, snapshots)["generation"]
            scheduler.step()
            if updates["update_norm"] == 0.0:
                raise AssertionError("B2 optimizer produced zero generation update")
            elapsed = time.time() - started
            record = {
                "step": step,
                "sample_id": str(row["sample_id"]),
                "image_id": int(row["image_id"]),
                "relation": str(row["relation"]),
                "prompt": str(row["prompt"]),
                "loss": float(loss.detach().cpu()),
                "raw_losses": {name: float(value.detach().cpu()) for name, value in raw_losses.items()},
                "weighted_losses": {name: float(value.detach().cpu()) for name, value in weighted.items()},
                "loss_weights": loss_weights,
                "grad_generation": raw_grad,
                "global_grad_norm": global_grad,
                "clipped_grad_generation": clipped_grad,
                "update_generation": updates["update_norm"],
                "relative_update_generation": updates["relative_update"],
                "lr_generation": applied_lr,
                "next_lr_generation": float(optimizer.param_groups[0]["lr"]),
                "flow_mse_readonly": float(flow_mse.cpu()),
                "teacher_flow_mse_readonly": teacher_flow_mse,
                "velocity_drift_readonly": velocity_drift,
                "frozen_parameter_gradients": 0,
                "details": details,
                "elapsed_seconds": elapsed,
            }
            records.append(record)
            status.update({
                "status": "running",
                "phase": "position_only",
                "step": step,
                "total_steps": args.max_steps,
                "percent": 100.0 * step / args.max_steps,
                "metrics": record,
                "elapsed_seconds": elapsed,
                "eta_seconds": elapsed / step * (args.max_steps - step),
                "updated_at": now_iso(),
            })
            atomic_json(status_path, status)
            atomic_json(args.run_dir / "bprime_position_metrics.json", {"records": records})
            print(
                f"step={step}/{args.max_steps} relation={row['relation']} "
                f"loss={record['loss']:.6f} grad={raw_grad:.4e} "
                f"update={updates['update_norm']:.4e} flow(readonly)={record['flow_mse_readonly']:.6f}",
                flush=True,
            )
            if step % args.save_every == 0 or step == args.max_steps:
                save_checkpoint(model, optimizer, scheduler, step, args.run_dir, record)

        status.update({
            "status": "completed",
            "phase": "completed",
            "percent": 100.0,
            "elapsed_seconds": time.time() - started,
            "updated_at": now_iso(),
        })
        atomic_json(status_path, status)
        atomic_json(args.run_dir / "metadata.json", {
            "status": "completed",
            "protocol": "bprime-position-only-real-coco-box-v1",
            "init_checkpoint": str(args.init_checkpoint.expanduser().resolve()),
            "init_checkpoint_step": init_step,
            "manifest": str(args.manifest.resolve()),
            "manifest_rows": len(dataset),
            "base_model_frozen": True,
            "frozen_modules": ["understanding", "relation_encoder", "Show-o2"],
            "trainable_module": "generation",
            "generation_adapter_policy": "single_generation_route",
            "forbidden_backward_losses": ["native_flow", "guided_flow", "velocity_retention", "mmu"],
            "position_losses": list(LOSS_NAMES),
            "loss_weights": loss_weights,
            "calibration_grad_norms": calibration_grad_norms,
            "lr_generation": args.lr_generation,
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
    finally:
        teacher_handle.remove()
        generation_handle.remove()


if __name__ == "__main__":
    raise SystemExit(main())
