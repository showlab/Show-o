#!/usr/bin/env python3
"""Run one no-optimizer K=4 TIFA cycle gradient and memory smoke."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.utils import image_transform
from evaluation.c4_drsa_adapter.train_drsa import encode_images
from evaluation.c5_boxdiff_adapter.staged_spatial_adapter_smoke import (
    build_generation_inputs,
    model_velocity,
)
from evaluation.c6_mask2former_alignment.frozen_u_cycle import (
    deterministic_image_latents,
    frozen_u_candidate_logits,
)
from evaluation.c6_mask2former_alignment.tifa_cycle import (
    TifaCocoCycleDataset,
    build_tifa_question_batch,
    option_token_ids,
)
from evaluation.c6_mask2former_alignment.tifa_cycle_loss import (
    per_question_tifa_cycle_loss,
)
from evaluation.c6_mask2former_alignment.tifa_cycle_sampling import (
    prepare_spatial_cycle_questions,
)
from evaluation.c6_mask2former_alignment.trajectory_rollout import (
    differentiable_euler_rollout,
    select_rollout_times,
)
from guidance.sampler import make_time_grid
from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
from models.generation_adaptation import (
    checkpoint_model_kwargs,
    configure_from_checkpoint,
)
from models.misc import get_text_tokenizer
from transport import create_transport
from utils import path_to_llm_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--cycle-step", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=4)
    parser.add_argument("--min-non-spatial-dimensions", type=int, default=2)
    parser.add_argument("--min-valid-questions", type=int, default=2)
    parser.add_argument("--gate-margin", type=float, default=0.0)
    parser.add_argument("--max-text-tokens", type=int, default=96)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--schedule-points", type=int, default=50)
    parser.add_argument("--time-shifting-factor", type=float, default=3.0)
    parser.add_argument("--t-min", type=float, default=0.10)
    parser.add_argument("--t-max", type=float, default=0.80)
    parser.add_argument("--model", default="showlab/show-o2-1.5B")
    parser.add_argument("--vae-path", type=Path, default=ROOT / "Wan2.1_VAE.pth")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def decode_prediction(vae: WanVAE, latents: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=vae.dtype):
        decoded = vae.model.decode(latents.unsqueeze(2), vae.scale).float()
    return decoded.squeeze(2).clamp(-1.0, 1.0)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("K=4 TIFA cycle smoke requires CUDA")
    if args.rollout_steps != 4:
        raise ValueError("this smoke requires exactly four velocity evaluations")
    if args.schedule_points < args.rollout_steps:
        raise ValueError("schedule_points must be at least rollout_steps")
    if not 0.0 <= args.t_min < args.t_max <= 1.0:
        raise ValueError("expected 0 <= t_min < t_max <= 1")
    if args.gate_margin < 0.0:
        raise ValueError("gate_margin must be non-negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device, dtype = torch.device(args.device), torch.bfloat16

    dataset = TifaCocoCycleDataset(
        args.manifest,
        image_transform=lambda image: image_transform(image, resolution=432),
    )
    row = dataset[args.index % len(dataset)]
    qa_pairs = prepare_spatial_cycle_questions(
        row,
        max_questions=args.max_questions,
        min_non_spatial_dimensions=args.min_non_spatial_dimensions,
        base_seed=args.seed,
        cycle_step=args.cycle_step,
    )

    llm_path = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer, token_ids = get_text_tokenizer(
        llm_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[llm_path],
    )
    sequence_length = args.max_text_tokens + 734
    u_batch = build_tifa_question_batch(
        qa_pairs,
        tokenizer,
        token_ids,
        max_text_tokens=args.max_text_tokens,
        sequence_length=sequence_length,
        device=device,
    )
    candidate_ids = option_token_ids(tokenizer)
    generation_inputs = build_generation_inputs(
        str(row["caption"]),
        tokenizer,
        token_ids,
        args.max_text_tokens,
        sequence_length,
        device,
        dtype,
    )

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Showo2Qwen2_5.from_pretrained(
        args.model,
        use_safetensors=False,
        local_files_only=True,
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
        **checkpoint_model_kwargs(payload),
    )
    model.to(device=device, dtype=dtype).requires_grad_(False).eval()
    adaptation = configure_from_checkpoint(model, payload)
    adaptation.requires_grad_(True)
    adaptation.train()
    adaptation_parameters = tuple(adaptation.parameters())
    active_ids = {id(parameter) for parameter in adaptation_parameters}
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in active_ids
    ]
    if unexpected:
        raise AssertionError(f"unexpected trainable parameters: {unexpected[:5]}")

    vae = WanVAE(vae_pth=str(args.vae_path), dtype=dtype, device=device)
    vae.model.requires_grad_(False).eval()
    clean_latent = encode_images(vae, row["image"].unsqueeze(0), dtype)

    transport = create_transport(
        path_type="Linear",
        prediction="velocity",
        snr_type="lognorm",
        do_shift=True,
        seq_len=729,
    )
    time_grid = make_time_grid(
        transport,
        args.schedule_points,
        device,
        args.time_shifting_factor,
        reverse=False,
    )
    rollout_generator = torch.Generator(device="cpu").manual_seed(args.seed + 41)
    rollout_times = select_rollout_times(
        time_grid,
        args.rollout_steps,
        args.t_min,
        args.t_max,
        generator=rollout_generator,
    )
    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 73)
    noise = torch.randn(
        clean_latent.shape,
        generator=noise_generator,
        device=device,
        dtype=dtype,
    )
    _, noisy_latent, _ = transport.path_sampler.plan(
        rollout_times[:1], noise, clean_latent
    )

    with adaptation.enabled(False), torch.no_grad():
        teacher_logits = frozen_u_candidate_logits(
            model,
            clean_latent,
            u_batch,
            candidate_ids,
            sequence_length=sequence_length,
            dtype=dtype,
            device=device,
            attention_mask_builder=omni_attn_mask_naive,
            relation_codes=None,
        )

    def velocity_fn(
        state: torch.Tensor, current_timestep: torch.Tensor
    ) -> torch.Tensor:
        velocity, _ = model_velocity(
            model,
            generation_inputs,
            state,
            current_timestep,
            None,
            sequence_length,
            device,
        )
        return velocity

    torch.cuda.empty_cache()
    memory_before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    rollout = differentiable_euler_rollout(
        noisy_latent, rollout_times, velocity_fn
    )
    predicted_clean = rollout.terminal_clean
    predicted_image = decode_prediction(vae, predicted_clean)
    predicted_u_latent = deterministic_image_latents(vae, predicted_image, dtype)

    with adaptation.enabled(False):
        student_logits = frozen_u_candidate_logits(
            model,
            predicted_u_latent,
            u_batch,
            candidate_ids,
            sequence_length=sequence_length,
            dtype=dtype,
            device=device,
            attention_mask_builder=omni_attn_mask_naive,
            relation_codes=None,
        )
    loss, gate_details = per_question_tifa_cycle_loss(
        student_logits,
        teacher_logits,
        u_batch.targets,
        u_batch.candidate_mask,
        gate_margin=args.gate_margin,
        min_valid_questions=args.min_valid_questions,
    )
    gradients = torch.autograd.grad(
        loss, adaptation_parameters, allow_unused=True
    )
    squares = [
        gradient.detach().float().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    gradient_norm = (
        float(torch.stack(squares).sum().sqrt().cpu()) if squares else 0.0
    )
    nonzero_tensors = sum(
        int(gradient is not None and bool(gradient.detach().ne(0).any()))
        for gradient in gradients
    )

    teacher_valid = list(gate_details["teacher_valid"])
    valid_dimensions = sorted({
        question.dimension
        for question, valid in zip(u_batch.questions, teacher_valid)
        if valid
    })
    spatial_valid = any(
        question.dimension == "spatial" and valid
        for question, valid in zip(u_batch.questions, teacher_valid)
    )
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    loss_value = float(loss.detach().cpu())
    checks = {
        "finite_positive_loss": bool(torch.isfinite(loss.detach()).item())
        and loss_value > 0.0,
        "step_gate_valid": bool(gate_details["step_valid"]),
        "spatial_teacher_gate_valid": spatial_valid,
        "at_least_two_valid_dimensions": len(valid_dimensions) >= 2,
        "student_input_requires_grad": bool(predicted_u_latent.requires_grad),
        "all_adaptation_tensors_nonzero": nonzero_tensors
        == len(adaptation_parameters),
        "four_velocity_evaluations": len(rollout.velocities) == 4,
        "three_recurrent_transitions": rollout.transition_count == 3,
    }
    masked_student = student_logits.float().masked_fill(
        ~u_batch.candidate_mask, float("-inf")
    )
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "label": args.label,
        "route": adaptation.route,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(payload.get("step", -1)),
        "sample_id": row["sample_id"],
        "caption": row["caption"],
        "questions": [
            {
                "dimension": question.dimension,
                "question": question.question,
                "choices": list(question.choices),
                "target": question.target,
                "teacher_valid": teacher_valid[index],
            }
            for index, question in enumerate(u_batch.questions)
        ],
        "understanding_input": "predicted_image+question+choices_only",
        "generation_caption_excluded_from_u": True,
        "rollout": {
            "k": len(rollout.velocities),
            "schedule_points": args.schedule_points,
            "time_shifting_factor": args.time_shifting_factor,
            "times": [
                float(value) for value in rollout.times.detach().cpu()
            ],
            "model_evaluations": len(rollout.velocities),
            "recurrent_transitions": rollout.transition_count,
        },
        "loss": loss_value,
        "gate": gate_details,
        "student_correct": masked_student.argmax(dim=-1).eq(
            u_batch.targets
        ).detach().cpu().tolist(),
        "valid_dimensions": valid_dimensions,
        "spatial_teacher_gate_valid": spatial_valid,
        "adaptation_gradient_norm": gradient_norm,
        "adaptation_gradient_tensors_nonzero": nonzero_tensors,
        "adaptation_parameter_tensors": len(adaptation_parameters),
        "decode_reencode_latent_mse": float(
            F.mse_loss(
                predicted_u_latent.float(), predicted_clean.detach().float()
            ).detach().cpu()
        ),
        "gpu_memory": {
            "allocated_before_bytes": memory_before,
            "peak_allocated_bytes": peak_allocated,
            "peak_increment_bytes": peak_allocated - memory_before,
            "peak_reserved_bytes": peak_reserved,
        },
        "checks": checks,
        "optimizer_created": False,
        "formal_training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
