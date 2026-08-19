#!/usr/bin/env python3
"""Train parameter-matched generation-only G-adapter or LoRA routes."""

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
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image

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
)
from evaluation.c5_boxdiff_adapter.staged_spatial_adapter_smoke import (
    build_generation_inputs,
    model_velocity,
)
from evaluation.c6_mask2former_alignment.online_detector_loss import (
    AUXILIARY_LOSS_NAMES,
    LOSS_NAMES,
    TRAINING_LOSS_NAMES,
    alignment_losses,
    class_conditioned_outputs,
    compose_training_loss,
)
from evaluation.c6_mask2former_alignment.frozen_u_cycle import (
    RELATION_OPTIONS,
    build_question_batch,
    build_relation_choice_batch,
    deterministic_image_latents,
    frozen_u_candidate_logits,
    prompt_u_task_loss,
    relation_option_token_ids,
    yes_no_token_ids,
)
from evaluation.c6_mask2former_alignment.trajectory_rollout import (
    differentiable_euler_rollout,
    select_rollout_times,
)
from guidance.sampler import make_time_grid
from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
from models.generation_adaptation import (
    ADAPTATION_ROUTES,
    GenerationAdaptation,
    configure_generation_adaptation,
)
from models.misc import get_text_tokenizer
from transport import create_transport
from utils import path_to_llm_name


CLASS_ALIASES = {
    "mouse": "computer mouse",
    "remote": "tv remote",
    "keyboard": "computer keyboard",
}


DEFAULT_WEIGHTS = {
    "mask_bce": 0.01,
    "mask_dice": 0.07,
    "presence": 0.16,
    "relation_distill": 0.5,
    "relation_pass": 1.0,
    "object_detect": 1.0,
}

DEFAULT_AUXILIARY_WEIGHTS = {
    "flow": 0.25,
    "teacher_retention": 0.20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, default=WORKSPACE / "data/drsa/spright_coco/data")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument(
        "--adaptation-route",
        choices=ADAPTATION_ROUTES,
        default="g_adapter",
    )
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=float)
    parser.add_argument("--parameter-match-tolerance", type=float, default=0.01)
    parser.add_argument("--model", default="showlab/show-o2-1.5B")
    parser.add_argument("--vae-path", type=Path, default=ROOT / "Wan2.1_VAE.pth")
    parser.add_argument(
        "--detector-config",
        type=Path,
        default=WORKSPACE / "mmdetection-2.28.2/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=WORKSPACE / "models/geneval/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth",
    )
    parser.add_argument(
        "--object-names",
        type=Path,
        default=WORKSPACE / "geneval/evaluation/object_names.txt",
    )
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument(
        "--save-steps",
        default="",
        help="Strictly increasing comma-separated checkpoint steps; overrides save-every.",
    )
    parser.add_argument("--preview-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument(
        "--lr-generation",
        "--lr-spatial",
        dest="lr_generation",
        type=float,
        default=1e-5,
        help="Learning rate for the selected generation-only adaptation route.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--detector-size", type=int, default=800)
    parser.add_argument("--query-temperature", type=float, default=0.25)
    parser.add_argument("--foreground-temperature", type=float, default=0.25)
    parser.add_argument("--object-threshold", type=float, default=0.3)
    parser.add_argument("--object-margin", type=float, default=0.05)
    parser.add_argument("--object-teacher-ratio", type=float, default=0.8)
    parser.add_argument(
        "--loss-ablation",
        choices=("original_mask_logits", "corrected", "teacher_validity_only"),
        default="original_mask_logits",
    )
    parser.add_argument("--t-min", type=float, default=0.10)
    parser.add_argument("--t-max", type=float, default=0.80)
    parser.add_argument("--rollout-steps", type=int, default=1)
    parser.add_argument("--rollout-schedule-points", type=int, default=50)
    parser.add_argument("--rollout-time-shifting-factor", type=float, default=3.0)
    parser.add_argument("--rollout-guidance-scale", type=float, default=0.0)
    parser.add_argument("--probe-component-grads", action="store_true")
    parser.add_argument("--weight-u-task", type=float, default=0.0)
    parser.add_argument(
        "--u-task-mode",
        choices=("relation_choice", "yes_no_triples"),
        default="relation_choice",
    )
    parser.add_argument("--u-cycle-ramp-steps", type=int, default=0)
    parser.add_argument(
        "--u-gate-policy",
        choices=("strict", "none"),
        default="strict",
    )
    parser.add_argument("--u-gate-margin", type=float, default=0.0)
    parser.add_argument(
        "--u-cycle-use-drsa-understanding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Legacy compatibility flag; generation-only training requires this to be false.",
    )
    for name, default in DEFAULT_WEIGHTS.items():
        parser.add_argument(f"--weight-{name.replace('_', '-')}", type=float, default=default)
    for name, default in DEFAULT_AUXILIARY_WEIGHTS.items():
        parser.add_argument(f"--weight-{name.replace('_', '-')}", type=float, default=default)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def detector_preprocess(images: torch.Tensor, size: int) -> torch.Tensor:
    """Differentiable equivalent of GenEval's RGB resize/normalization path."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"expected RGB images [B,3,H,W], got {tuple(images.shape)}")
    pixels = (images.float().clamp(-1.0, 1.0) + 1.0) * 127.5
    pixels = F.interpolate(pixels, size=(size, size), mode="bilinear", align_corners=False)
    mean = pixels.new_tensor([123.675, 116.28, 103.53])[None, :, None, None]
    std = pixels.new_tensor([58.395, 57.12, 57.375])[None, :, None, None]
    return (pixels - mean) / std


def detector_metas(batch: int, size: int) -> list[dict[str, Any]]:
    return [
        {
            "batch_input_shape": (size, size),
            "img_shape": (size, size, 3),
            "ori_shape": (size, size, 3),
            "pad_shape": (size, size, 3),
            "scale_factor": 1.0,
            "flip": False,
            "flip_direction": None,
        }
        for _ in range(batch)
    ]


def detector_raw(detector: torch.nn.Module, images: torch.Tensor, size: int):
    normalized = detector_preprocess(images, size)
    metas = detector_metas(len(images), size)
    features = detector.extract_feat(normalized)
    all_class_logits, all_mask_logits = detector.panoptic_head(features, metas)
    return all_class_logits[-1], all_mask_logits[-1]


def decode_prediction(vae: WanVAE, latents: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=vae.dtype):
        decoded = vae.model.decode(latents.unsqueeze(2), vae.scale).float()
    return decoded.squeeze(2).clamp(-1.0, 1.0)


def component_grad_norms(
    losses: Mapping[str, torch.Tensor], parameters: tuple[torch.nn.Parameter, ...]
) -> dict[str, float]:
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    output = {}
    names = list(losses)
    for index, name in enumerate(names):
        if not losses[name].requires_grad:
            output[name] = 0.0
            continue
        gradients = torch.autograd.grad(
            losses[name],
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        squares = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
        output[name] = float(torch.stack(squares).sum().sqrt().cpu()) if squares else 0.0
    return output


def adaptation_grad_norm(adaptation: GenerationAdaptation) -> float:
    parameters = tuple(adaptation.parameters())
    if not parameters:
        return 0.0
    device = parameters[0].device
    total = torch.zeros((), device=device, dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            total = total + parameter.grad.detach().float().square().sum()
    return float(total.sqrt().cpu())


def frozen_generation_teacher_velocity(
    model: Showo2Qwen2_5,
    adaptation: GenerationAdaptation,
    inputs: dict[str, torch.Tensor],
    noisy_latent: torch.Tensor,
    timestep: torch.Tensor,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Run unadapted Show-o2 as the frozen retention teacher."""
    with adaptation.enabled(False):
        with torch.no_grad():
            velocity, _ = model_velocity(
                model,
                inputs,
                noisy_latent,
                timestep,
                None,
                sequence_length,
                device,
            )
    return velocity.detach()


def save_adaptation_checkpoint(
    adaptation: GenerationAdaptation,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    step: int,
    run_dir: Path,
    metrics: Mapping[str, Any],
    *,
    training_generator: torch.Generator,
    rollout_generator: torch.Generator,
) -> Path:
    checkpoint = run_dir / "checkpoints" / f"step-{step:06d}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "step": step,
        "adaptation_config": adaptation.config_dict(),
        "adaptation_state": adaptation.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": dict(metrics),
        "rng_state": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(training_generator.device),
            "training_generator": training_generator.get_state(),
            "rollout_generator": rollout_generator.get_state(),
        },
    }, checkpoint)
    (run_dir / "latest_checkpoint.txt").write_text(str(checkpoint) + "\n")
    return checkpoint


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def main() -> int:
    args = parse_args()
    if args.max_steps < 1 or args.save_every < 1 or args.preview_every < 1:
        raise ValueError("step/save/preview intervals must be positive")
    try:
        save_steps = tuple(int(value) for value in args.save_steps.split(",") if value)
    except ValueError as error:
        raise ValueError("save-steps must contain comma-separated integers") from error
    if save_steps:
        if tuple(sorted(set(save_steps))) != save_steps:
            raise ValueError("save-steps must be unique and strictly increasing")
        if save_steps[0] < 1 or save_steps[-1] > args.max_steps:
            raise ValueError("save-steps must be in [1, max-steps]")
        if save_steps[-1] != args.max_steps:
            raise ValueError("explicit save-steps must include max-steps")
    effective_save_steps = set(save_steps)
    if not 0.0 <= args.t_min < args.t_max <= 1.0:
        raise ValueError("expected 0 <= t-min < t-max <= 1")
    if args.rollout_steps < 1:
        raise ValueError("rollout-steps must be positive")
    if args.rollout_schedule_points < 2:
        raise ValueError("rollout-schedule-points must be at least two")
    if args.rollout_time_shifting_factor <= 0.0:
        raise ValueError("rollout-time-shifting-factor must be positive")
    if args.rollout_guidance_scale != 0.0:
        raise ValueError(
            "K-step comparison currently requires rollout-guidance-scale=0"
        )
    if args.object_threshold < 0.0 or args.object_margin < 0.0:
        raise ValueError("object threshold and margin must be non-negative")
    if not 0.0 <= args.object_teacher_ratio <= 1.0:
        raise ValueError("object teacher ratio must be between zero and one")
    configured_weights = {
        name: float(getattr(args, f"weight_{name}"))
        for name in TRAINING_LOSS_NAMES
    }
    configured_weights["u_task"] = float(args.weight_u_task)
    if any(value < 0.0 for value in configured_weights.values()):
        raise ValueError("loss weights must be non-negative")
    if args.u_cycle_ramp_steps < 0:
        raise ValueError("u-cycle-ramp-steps must be non-negative")
    if args.u_gate_margin < 0.0:
        raise ValueError("u-gate-margin must be non-negative")
    if args.parameter_match_tolerance < 0.0:
        raise ValueError("parameter-match-tolerance must be non-negative")
    if args.u_cycle_use_drsa_understanding:
        raise ValueError(
            "generation-only training cannot use the legacy DRSA understanding route"
        )
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("online Mask2Former alignment requires CUDA")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "previews").mkdir(exist_ok=True)
    original_mask_logits = args.loss_ablation == "original_mask_logits"
    teacher_validity_mask = args.loss_ablation in ("corrected", "teacher_validity_only")
    quality_mode = (
        "smooth"
        if args.loss_ablation in ("original_mask_logits", "teacher_validity_only")
        else "official"
    )
    object_teacher_ratio = (
        0.0
        if args.loss_ablation in ("original_mask_logits", "teacher_validity_only")
        else args.object_teacher_ratio
    )
    object_loss_mode = (
        "linear"
        if args.loss_ablation in ("original_mask_logits", "teacher_validity_only")
        else "squared"
    )
    status_path = args.run_dir / "status.json"
    metrics_path = args.run_dir / "metrics.jsonl"
    status: dict[str, Any] = {
        "status": "initializing",
        "phase": "loading",
        "step": 0,
        "total_steps": args.max_steps,
        "percent": 0.0,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "run_dir": str(args.run_dir.resolve()),
    }
    atomic_json(status_path, status)
    started = time.time()
    try:
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
        u_candidate_token_ids = (
            relation_option_token_ids(tokenizer)
            if args.u_task_mode == "relation_choice"
            else yes_no_token_ids(tokenizer)
        )
        model = Showo2Qwen2_5.from_pretrained(
            args.model,
            use_safetensors=False,
            local_files_only=True,
            low_cpu_mem_usage=False,
            torch_dtype=dtype,
            drsa_enabled=False,
            generation_adapter_enabled=args.adaptation_route == "g_adapter",
        )
        model.to(device=device, dtype=dtype).requires_grad_(False)
        adaptation = configure_generation_adaptation(
            model,
            args.adaptation_route,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            parameter_match_tolerance=args.parameter_match_tolerance,
        )
        resume_payload = None
        start_step = 0
        elapsed_offset = 0.0
        if args.resume_checkpoint is not None:
            resume_path = args.resume_checkpoint.expanduser().resolve()
            if not resume_path.is_file():
                raise FileNotFoundError(resume_path)
            resume_payload = torch.load(
                resume_path, map_location="cpu", weights_only=True
            )
            resume_config = resume_payload.get("adaptation_config", {})
            if resume_config.get("route") != args.adaptation_route:
                raise ValueError(
                    f"resume route {resume_config.get('route')!r} does not match "
                    f"--adaptation-route={args.adaptation_route!r}"
                )
            adaptation.load_state_dict(resume_payload["adaptation_state"], strict=True)
            start_step = int(resume_payload.get("step", 0))
            elapsed_offset = float(
                resume_payload.get("metrics", {}).get("elapsed_seconds", 0.0)
            )
            if start_step < 1 or start_step > args.max_steps:
                raise ValueError(
                    f"resume step {start_step} must be in [1, {args.max_steps}]"
                )
        model.requires_grad_(False)
        adaptation.requires_grad_(True)
        model.eval()
        adaptation.train()
        adaptation_parameters = tuple(adaptation.parameters())
        active_names = [name for name, _ in adaptation.named_parameters()]
        model_active = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        model_active_names = [name for name, _ in model_active]
        if (
            not active_names
            or {id(parameter) for _, parameter in model_active}
            != {id(parameter) for parameter in adaptation_parameters}
        ):
            raise AssertionError(
                f"unexpected trainable parameters: route={active_names[:5]} "
                f"model={model_active_names[:5]}"
            )

        vae = WanVAE(vae_pth=str(args.vae_path), dtype=dtype, device=device)
        vae.model.requires_grad_(False).eval()
        from mmdet.apis import init_detector

        detector = init_detector(
            str(args.detector_config), str(args.detector_checkpoint), device=str(device)
        )
        detector.eval().requires_grad_(False)
        if any(parameter.requires_grad for parameter in detector.parameters()):
            raise AssertionError("Mask2Former parameters must remain frozen")

        dataset = BprimeCocoBoxDataset(
            args.manifest,
            args.shard_root,
            image_transform=partial(image_transform, resolution=432),
        )
        if args.max_steps > len(dataset):
            raise ValueError(f"max-steps {args.max_steps} exceeds manifest rows {len(dataset)}")
        order = list(range(len(dataset)))
        random.Random(args.seed).shuffle(order)
        class_names = [line.strip() for line in args.object_names.read_text().splitlines() if line.strip()]
        class_to_id = {name: index for index, name in enumerate(class_names)}
        transport = create_transport(
            path_type="Linear", prediction="velocity", snr_type="lognorm", do_shift=True, seq_len=729
        )
        rollout_time_grid = make_time_grid(
            transport,
            args.rollout_schedule_points,
            device,
            args.rollout_time_shifting_factor,
            reverse=False,
        )
        rollout_generator = torch.Generator(device="cpu")
        rollout_generator.manual_seed(args.seed + 41)
        training_generator = torch.Generator(device=device)
        training_generator.manual_seed(args.seed + 73)
        optimizer = AdamW(
            [{"params": adaptation_parameters, "lr": args.lr_generation, "name": args.adaptation_route}],
            weight_decay=args.weight_decay,
        )
        warmup = max(1, min(args.warmup_steps, args.max_steps))
        scheduler = LambdaLR(
            optimizer,
            lambda step: float(step + 1) / warmup if step < warmup else 1.0,
        )
        if resume_payload is not None:
            optimizer.load_state_dict(resume_payload["optimizer"])
            scheduler.load_state_dict(resume_payload["scheduler"])
            rng_state = resume_payload.get("rng_state")
            if not isinstance(rng_state, Mapping):
                raise ValueError("resume checkpoint has no complete rng_state")
            random.setstate(rng_state["python"])
            torch.set_rng_state(rng_state["torch_cpu"])
            torch.cuda.set_rng_state(rng_state["torch_cuda"], device=device)
            training_generator.set_state(rng_state["training_generator"])
            rollout_generator.set_state(rng_state["rollout_generator"])
            del resume_payload
        detector_weights = {
            name: float(getattr(args, f"weight_{name}")) for name in LOSS_NAMES
        }
        if original_mask_logits:
            detector_weights["object_detect"] = 0.0
        auxiliary_weights = {
            name: float(getattr(args, f"weight_{name}"))
            for name in AUXILIARY_LOSS_NAMES
        }
        weights = {**detector_weights, **auxiliary_weights}
        reported_weights = {**weights, "u_task": float(args.weight_u_task)}
        active_losses = [
            name for name, value in reported_weights.items() if value != 0.0
        ]
        u_cycle_enabled = args.weight_u_task > 0.0
        u_gate_valid_count = 0
        u_gate_total_count = 0
        status.update({
            "status": "running",
            "phase": "online_detector_alignment",
            "manifest": str(args.manifest.resolve()),
            "seed": args.seed,
            "manifest_rows": len(dataset),
            "initialization": "original_showo2_zero_output_adaptation",
            "resume_checkpoint": (
                str(args.resume_checkpoint.expanduser().resolve())
                if args.resume_checkpoint is not None else None
            ),
            "resume_step": start_step,
            "detector_checkpoint": str(args.detector_checkpoint.resolve()),
            "detector_size": args.detector_size,
            "adaptation": adaptation.config_dict(),
            "trainable_parameters": adaptation.parameter_count,
            "trainable_parameter_names": active_names,
            "generation_adapter_policy": "parameter_matched_generation_only",
            "paired_training_rng_seed": args.seed + 73,
            "loss_weights": reported_weights,
            "teacher_replay": {
                "enabled": auxiliary_weights["teacher_retention"] > 0.0,
                "loss_key": "teacher_retention",
                "caption_field": "original_caption",
                "student": "active generation adaptation",
                "teacher": "same frozen Show-o2 with generation adaptation disabled",
                "matching": "velocity MSE at identical noisy latent and timestep",
            },
            "detector_loss_weights": detector_weights,
            "auxiliary_loss_weights": auxiliary_weights,
            "loss_ablation": args.loss_ablation,
            "checkpoint_schedule": {
                "mode": "explicit" if save_steps else "interval",
                "save_every": None if save_steps else args.save_every,
                "save_steps": list(save_steps),
            },
            "detector_quality_mode": quality_mode,
            "teacher_validity_mask": teacher_validity_mask,
            "object_loss_mode": object_loss_mode,
            "object_teacher_ratio": object_teacher_ratio,
            "object_detection_threshold": args.object_threshold,
            "object_detection_margin": args.object_margin,
            "invalid_teacher_policy": (
                "none" if original_mask_logits else "mask_teacher_dependent_losses"
            ),
            "u_cycle": {
                "enabled": u_cycle_enabled,
                "objective": args.u_task_mode,
                "image_path": "predicted_clean_decode_deterministic_reencode_frozen_u",
                "answer_image_role": "teacher_validity_only",
                "gate_policy": args.u_gate_policy,
                "gate_margin": args.u_gate_margin,
                "ramp_steps": args.u_cycle_ramp_steps,
                "uses_frozen_drsa_understanding": False,
                "generation_adaptation_disabled_during_u": True,
                "candidate_token_ids": list(u_candidate_token_ids),
            },
            "timestep_range": [args.t_min, args.t_max],
            "rollout": {
                "model_evaluations": args.rollout_steps,
                "recurrent_euler_transitions": max(0, args.rollout_steps - 1),
                "schedule_points": args.rollout_schedule_points,
                "time_shifting_factor": args.rollout_time_shifting_factor,
                "guidance_scale": args.rollout_guidance_scale,
                "loss_reduction": "mean_over_velocity_evaluations",
                "image_supervision": "terminal_predicted_clean",
            },
            "updated_at": now_iso(),
        })
        atomic_json(status_path, status)
        torch.cuda.reset_peak_memory_stats(device)

        for step in range(start_step + 1, args.max_steps + 1):
            row = dataset[order[step - 1]]
            relations = tuple(str(value) for value in row["relations"])
            relation = "+".join(relations)
            phrases = [str(item["phrase"]) for item in row["objects"]]
            try:
                class_ids = [class_to_id[CLASS_ALIASES.get(phrase, phrase)] for phrase in phrases]
            except KeyError as error:
                raise KeyError(f"Mask2Former has no COCO class for {phrases}") from error
            inputs = build_generation_inputs(
                str(row["prompt"]), tokenizer, token_ids, args.max_text_tokens,
                sequence_length, device, dtype,
            )
            u_batch = None
            if u_cycle_enabled:
                if len(relations) != 1:
                    raise ValueError(
                        "frozen-U supervision is single-axis only; "
                        f"sample {row['sample_id']} has relations={relations}"
                    )
                question_builder = (
                    build_relation_choice_batch
                    if args.u_task_mode == "relation_choice"
                    else build_question_batch
                )
                u_batch = question_builder(
                    phrases, relations[0], tokenizer, token_ids,
                    max_text_tokens=args.max_text_tokens,
                    sequence_length=sequence_length, device=device,
                )
            original_caption = str(row.get("original_caption", "")).strip()
            if auxiliary_weights["teacher_retention"] > 0.0 and not original_caption:
                raise ValueError(
                    f"sample {row['sample_id']} has no original_caption for teacher retention"
                )
            native_inputs = None
            if auxiliary_weights["teacher_retention"] > 0.0:
                native_inputs = build_generation_inputs(
                    original_caption,
                    tokenizer,
                    token_ids,
                    args.max_text_tokens,
                    sequence_length,
                    device,
                    dtype,
                )

            real_image = row["image"].unsqueeze(0).to(device=device, dtype=dtype)
            clean_latent = encode_images(vae, row["image"].unsqueeze(0), dtype)
            noise = torch.randn(
                clean_latent.shape,
                generator=training_generator,
                device=clean_latent.device,
                dtype=clean_latent.dtype,
            )
            if args.rollout_steps == 1:
                timestep = torch.empty(
                    (1,), device=device, dtype=dtype
                ).uniform_(
                    args.t_min, args.t_max, generator=training_generator
                )
                rollout_times = timestep.float()
            else:
                rollout_times = select_rollout_times(
                    rollout_time_grid,
                    args.rollout_steps,
                    args.t_min,
                    args.t_max,
                    generator=rollout_generator,
                )
                timestep = rollout_times[:1]
            _, noisy_latent, velocity_target = transport.path_sampler.plan(
                timestep, noise, clean_latent
            )

            teacher_u_logits = None
            if u_cycle_enabled:
                assert u_batch is not None
                with adaptation.enabled(False), torch.no_grad():
                    teacher_u_logits = frozen_u_candidate_logits(
                        model,
                        clean_latent,
                        u_batch,
                        u_candidate_token_ids,
                        sequence_length=sequence_length,
                        dtype=dtype,
                        device=device,
                        attention_mask_builder=omni_attn_mask_naive,
                        relation_codes=None,
                    )

            with torch.no_grad():
                teacher_cls, teacher_masks = detector_raw(
                    detector, real_image.float(), args.detector_size
                )
                teacher = class_conditioned_outputs(
                    teacher_cls,
                    teacher_masks,
                    class_ids,
                    query_temperature=args.query_temperature,
                    foreground_temperature=args.foreground_temperature,
                    quality_mode=quality_mode,
                )
                teacher = {key: value.detach() for key, value in teacher.items()}
            del teacher_cls, teacher_masks

            optimizer.zero_grad(set_to_none=True)
            def prompt_velocity_fn(
                state: torch.Tensor, current_timestep: torch.Tensor
            ) -> torch.Tensor:
                result, _ = model_velocity(
                    model,
                    inputs,
                    state,
                    current_timestep,
                    None,
                    sequence_length,
                    device,
                )
                return result

            prompt_rollout = differentiable_euler_rollout(
                noisy_latent, rollout_times, prompt_velocity_fn
            )
            velocity = prompt_rollout.velocities[-1]
            predicted_clean = prompt_rollout.terminal_clean
            predicted_image = decode_prediction(vae, predicted_clean)
            predicted_u_latent = None
            u_task_details = None
            if u_cycle_enabled:
                assert u_batch is not None and teacher_u_logits is not None
                predicted_u_latent = deterministic_image_latents(
                    vae, predicted_image, dtype
                )
                with adaptation.enabled(False):
                    student_u_logits = frozen_u_candidate_logits(
                        model,
                        predicted_u_latent,
                        u_batch,
                        u_candidate_token_ids,
                        sequence_length=sequence_length,
                        dtype=dtype,
                        device=device,
                        attention_mask_builder=omni_attn_mask_naive,
                        relation_codes=None,
                    )
                u_task, u_task_details = prompt_u_task_loss(
                    student_u_logits,
                    teacher_u_logits,
                    u_batch.targets,
                    gate_policy=args.u_gate_policy,
                    gate_margin=args.u_gate_margin,
                )
                u_task_details["questions"] = [
                    {
                        "name": question.name,
                        "statement": question.statement,
                        "relation": question.relation,
                        "target": (
                            RELATION_OPTIONS[question.target]
                            if args.u_task_mode == "relation_choice"
                            else "yes" if question.target == 0 else "no"
                        ),
                    }
                    for question in u_batch.questions
                ]
                u_task_details["decode_reencode_latent_mse"] = float(
                    F.mse_loss(
                        predicted_u_latent.float(), predicted_clean.detach().float()
                    ).detach().cpu()
                )
                u_task_details["student_input_requires_grad"] = bool(
                    predicted_u_latent.requires_grad
                )
                u_gate_total_count += 1
                u_gate_valid_count += int(u_task_details["teacher_strict_valid"])
            else:
                student_u_logits = None
                u_task = velocity.new_zeros((), dtype=torch.float32)
            student_cls, student_masks = detector_raw(
                detector, predicted_image, args.detector_size
            )
            student = class_conditioned_outputs(
                student_cls,
                student_masks,
                class_ids,
                query_temperature=args.query_temperature,
                foreground_temperature=args.foreground_temperature,
                quality_mode=quality_mode,
            )
            detector_losses, details = alignment_losses(
                student,
                teacher,
                [relations],
                object_threshold=args.object_threshold,
                object_margin=args.object_margin,
                object_teacher_ratio=object_teacher_ratio,
                teacher_validity_mask=teacher_validity_mask,
                object_loss_mode=object_loss_mode,
            )
            flow_step_losses = tuple(
                F.mse_loss(current_velocity.float(), velocity_target.float())
                for current_velocity in prompt_rollout.velocities
            )
            flow = torch.stack(flow_step_losses).mean()
            native_rollout = None
            native_velocity = None
            teacher_velocity = None
            teacher_velocities: list[torch.Tensor] = []
            teacher_retention_steps: tuple[torch.Tensor, ...] = ()
            if auxiliary_weights["teacher_retention"] > 0.0:
                assert native_inputs is not None
                def native_velocity_fn(
                    state: torch.Tensor, current_timestep: torch.Tensor
                ) -> torch.Tensor:
                    result, _ = model_velocity(
                        model,
                        native_inputs,
                        state,
                        current_timestep,
                        None,
                        sequence_length,
                        device,
                    )
                    teacher_result = frozen_generation_teacher_velocity(
                        model,
                        adaptation,
                        native_inputs,
                        state,
                        current_timestep,
                        sequence_length,
                        device,
                    )
                    teacher_velocities.append(teacher_result)
                    return result

                native_rollout = differentiable_euler_rollout(
                    noisy_latent, rollout_times, native_velocity_fn
                )
                native_velocity = native_rollout.velocities[-1]
                teacher_velocity = teacher_velocities[-1]
                teacher_retention_steps = tuple(
                    F.mse_loss(active.float(), frozen.float())
                    for active, frozen in zip(
                        native_rollout.velocities, teacher_velocities
                    )
                )
                teacher_retention = torch.stack(teacher_retention_steps).mean()
            else:
                teacher_retention = velocity.new_zeros((), dtype=torch.float32)
            base_losses = {
                **detector_losses,
                "flow": flow,
                "teacher_retention": teacher_retention,
            }
            loss, weighted_losses = compose_training_loss(base_losses, weights)
            u_ramp = (
                min(1.0, step / args.u_cycle_ramp_steps)
                if args.u_cycle_ramp_steps > 0
                else 1.0
            )
            effective_u_weight = float(args.weight_u_task) * u_ramp
            weighted_losses["u_task"] = u_task.float() * effective_u_weight
            loss = loss + weighted_losses["u_task"]
            raw_losses = {**base_losses, "u_task": u_task}
            per_component = None
            if args.probe_component_grads and step == 1:
                per_component = component_grad_norms(raw_losses, adaptation_parameters)
            loss.backward()
            loss_value = float(loss.detach().cpu())
            raw_grad = adaptation_grad_norm(adaptation)
            if not math.isfinite(raw_grad):
                raise FloatingPointError(f"non-finite generation gradient {raw_grad}")
            if raw_grad <= 0.0 and loss_value > 1e-8:
                raise FloatingPointError(
                    f"positive loss {loss_value} has no generation gradient"
                )
            optimizer_skipped = raw_grad <= 0.0
            frozen_gradients = [
                name for name, parameter in model.named_parameters()
                if not parameter.requires_grad and parameter.grad is not None
            ]
            detector_gradients = [
                name for name, parameter in detector.named_parameters() if parameter.grad is not None
            ]
            vae_gradients = [
                name for name, parameter in vae.model.named_parameters()
                if parameter.grad is not None
            ]
            if frozen_gradients or detector_gradients or vae_gradients:
                raise AssertionError(
                    f"frozen gradients model={frozen_gradients[:3]} "
                    f"detector={detector_gradients[:3]} vae={vae_gradients[:3]}"
                )
            applied_lr = float(optimizer.param_groups[0]["lr"])
            if optimizer_skipped:
                global_grad = 0.0
            else:
                global_grad = float(
                    torch.nn.utils.clip_grad_norm_(
                        adaptation_parameters, args.grad_clip
                    ).cpu()
                )
                optimizer.step()
                scheduler.step()

            elapsed = elapsed_offset + time.time() - started
            teacher_object_valid = bool(details["teacher_object_valid"][0])
            teacher_relation_valid = bool(details["teacher_relation_valid"][0])
            teacher_valid = bool(details["teacher_valid"][0])
            record = {
                "step": step,
                "sample_id": str(row["sample_id"]),
                "image_id": int(row["image_id"]),
                "prompt": str(row["prompt"]),
                "original_caption": original_caption,
                "phrases": phrases,
                "class_ids": class_ids,
                "relation": str(row["relation"]),
                "relations": list(relations),
                "timestep": float(timestep.item()),
                "rollout_times": [
                    float(value) for value in rollout_times.detach().cpu()
                ],
                "rollout_model_evaluations": len(prompt_rollout.velocities),
                "rollout_recurrent_transitions": prompt_rollout.transition_count,
                "loss": loss_value,
                "optimizer_skipped": optimizer_skipped,
                "raw_losses": {name: float(value.detach().cpu()) for name, value in raw_losses.items()},
                "weighted_losses": {
                    name: float(value.detach().cpu()) for name, value in weighted_losses.items()
                },
                "component_grad_norms": per_component,
                "grad_generation": raw_grad,
                "global_grad_norm": global_grad,
                "lr_generation": applied_lr,
                "flow_mse": float(flow.detach().cpu()),
                "flow_step_mse": [
                    float(value.detach().cpu()) for value in flow_step_losses
                ],
                "teacher_retention_mse": float(teacher_retention.detach().cpu()),
                "teacher_retention_step_mse": [
                    float(value.detach().cpu())
                    for value in teacher_retention_steps
                ],
                "u_task": u_task_details,
                "u_task_effective_weight": effective_u_weight,
                "u_gate_coverage": (
                    u_gate_valid_count / u_gate_total_count
                    if u_gate_total_count else None
                ),
                "native_flow_mse": (
                    float(F.mse_loss(
                        native_velocity.detach().float(), velocity_target.float()
                    ).cpu())
                    if native_velocity is not None else None
                ),
                "teacher_flow_mse": (
                    float(F.mse_loss(
                        teacher_velocity.float(), velocity_target.float()
                    ).cpu())
                    if teacher_velocity is not None else None
                ),
                "teacher_object_valid": teacher_object_valid,
                "teacher_relation_valid": teacher_relation_valid,
                "teacher_valid": teacher_valid,
                "details": details,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
                "elapsed_seconds": elapsed,
            }
            append_jsonl(metrics_path, record)
            if step == 1 or step % args.preview_every == 0 or step == args.max_steps:
                preview = torch.cat((real_image.float(), predicted_image.detach()), dim=0)
                save_image(
                    (preview + 1.0) * 0.5,
                    args.run_dir / "previews" / f"step-{step:06d}-real-pred.png",
                    nrow=2,
                )
            should_save = (
                step in effective_save_steps
                if save_steps
                else step % args.save_every == 0 or step == args.max_steps
            )
            if should_save:
                save_adaptation_checkpoint(
                    adaptation,
                    optimizer,
                    scheduler,
                    step,
                    args.run_dir,
                    record,
                    training_generator=training_generator,
                    rollout_generator=rollout_generator,
                )

            status.update({
                "status": "running",
                "phase": "online_detector_alignment",
                "step": step,
                "total_steps": args.max_steps,
                "percent": 100.0 * step / args.max_steps,
                "metrics": record,
                "elapsed_seconds": elapsed,
                "eta_seconds": elapsed / step * (args.max_steps - step),
                "updated_at": now_iso(),
            })
            atomic_json(status_path, status)
            print(
                f"step={step}/{args.max_steps} relation={relation} loss={record['loss']:.6f} "
                f"score_s={details['student_relation_score'][0]:.4f} "
                f"score_t={details['teacher_relation_score'][0]:.4f} grad={raw_grad:.4e} "
                f"flow={record['flow_mse']:.4f} ret={record['teacher_retention_mse']:.4e} "
                f"u={float(u_task.detach().cpu()):.4f} "
                f"u_gate={u_gate_valid_count}/{u_gate_total_count} "
                f"mem={record['peak_cuda_memory_bytes'] / 2**30:.2f}GiB",
                flush=True,
            )
            del (
                student_cls,
                student_masks,
                student,
                teacher,
                predicted_image,
                predicted_clean,
                native_velocity,
                teacher_velocity,
                predicted_u_latent,
                student_u_logits,
                teacher_u_logits,
            )

        status.update({
            "status": "completed",
            "phase": "completed",
            "percent": 100.0,
            "elapsed_seconds": time.time() - started,
            "updated_at": now_iso(),
        })
        atomic_json(status_path, status)
        atomic_json(args.run_dir / "metadata.json", {
            "protocol": (
                "showo2-prompt-conditioned-frozen-u-cycle-v1"
                if u_cycle_enabled
                else "showo2-online-mask2former-multiaxis-v3"
            ),
            "manifest": str(args.manifest.resolve()),
            "seed": args.seed,
            "samples": args.max_steps,
            "mask_teacher": "same frozen official GenEval Mask2Former, detached real-image branch",
            "velocity_teacher": "unadapted frozen Show-o2, same native caption/x_t/t",
            "teacher_replay": {
                "enabled": auxiliary_weights["teacher_retention"] > 0.0,
                "loss_key": "teacher_retention",
                "caption_field": "original_caption",
                "student": "active generation adaptation",
                "teacher": "same frozen Show-o2 with generation adaptation disabled",
                "matching": "velocity MSE at identical noisy latent and timestep",
            },
            "student": "differentiable predicted-x1 image branch",
            "relation_supervision": {
                "schema": "orthogonal-multilabel-v1",
                "maximum_targets_per_sample": 2,
                "same_axis_relations_are_mutually_exclusive": True,
                "orthogonal_relations_may_coexist": True,
                "per_sample_relation_reduction": "mean",
            },
            "u_student": (
                "predicted-x1 decode, deterministic VAE re-encode, frozen U/MMU"
                if u_cycle_enabled else None
            ),
            "u_teacher_gate": (
                (
                    "answer-image four-way relation-choice frozen-U strict validity"
                    if args.u_task_mode == "relation_choice"
                    else "answer-image positive/inverse/swap frozen-U strict validity"
                )
                if u_cycle_enabled else None
            ),
            "losses": active_losses,
            "loss_weights": reported_weights,
            "checkpoint_schedule": {
                "mode": "explicit" if save_steps else "interval",
                "save_every": None if save_steps else args.save_every,
                "save_steps": list(save_steps),
            },
            "base_model_frozen": True,
            "detector_frozen": True,
            "vae_frozen": True,
            "u_frozen": True,
            "adaptation": adaptation.config_dict(),
            "trainable_module": args.adaptation_route,
            "generation_adapter_policy": "parameter_matched_generation_only",
            "relation_encoder": None,
            "understanding_adapter": None,
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
