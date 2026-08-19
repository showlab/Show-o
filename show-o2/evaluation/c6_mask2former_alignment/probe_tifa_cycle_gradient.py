#!/usr/bin/env python3
"""Probe TIFA next-token gradients through a predicted image into one G route."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parents[1]
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
    select_multidimensional_questions,
    tifa_next_token_loss,
)
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
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-questions", type=int, default=3)
    parser.add_argument("--max-text-tokens", type=int, default=192)
    parser.add_argument("--timestep", type=float, default=0.5)
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
        raise RuntimeError("TIFA cycle gradient probe requires CUDA")
    if not 0.0 < args.timestep < 1.0:
        raise ValueError("timestep must be in (0,1)")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device, dtype = torch.device(args.device), torch.bfloat16
    dataset = TifaCocoCycleDataset(
        args.manifest,
        image_transform=lambda image: image_transform(image, resolution=432),
    )
    row = dataset[args.index % len(dataset)]
    qa_pairs = select_multidimensional_questions(
        row["qa_pairs"], max_questions=args.max_questions, seed=args.seed
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
        raise AssertionError(f"unexpected trainable model parameters: {unexpected[:5]}")

    vae = WanVAE(vae_pth=str(args.vae_path), dtype=dtype, device=device)
    vae.model.requires_grad_(False).eval()
    image = row["image"].unsqueeze(0).to(device=device, dtype=dtype)
    clean_latent = encode_images(vae, row["image"].unsqueeze(0), dtype)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    noise = torch.randn(
        clean_latent.shape, generator=generator, device=device, dtype=dtype
    )
    timestep = torch.tensor([args.timestep], device=device, dtype=dtype)
    transport = create_transport(
        path_type="Linear", prediction="velocity", snr_type="lognorm", do_shift=True,
        seq_len=729,
    )
    _, noisy_latent, _ = transport.path_sampler.plan(timestep, noise, clean_latent)
    generation_inputs = build_generation_inputs(
        str(row["caption"]), tokenizer, token_ids, args.max_text_tokens,
        sequence_length, device, dtype,
    )
    velocity, _ = model_velocity(
        model, generation_inputs, noisy_latent, timestep, None,
        sequence_length, device,
    )
    predicted_clean = noisy_latent.float() + (1.0 - timestep.float()).view(
        1, 1, 1, 1
    ) * velocity.float()
    predicted_image = decode_prediction(vae, predicted_clean)
    predicted_u_latent = deterministic_image_latents(vae, predicted_image, dtype)

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
    loss, details = tifa_next_token_loss(
        student_logits,
        u_batch.targets,
        u_batch.candidate_mask,
        gate_policy="none",
    )
    _, strict_details = tifa_next_token_loss(
        student_logits,
        u_batch.targets,
        u_batch.candidate_mask,
        teacher_logits=teacher_logits,
        gate_policy="strict",
    )
    gradients = torch.autograd.grad(loss, adaptation_parameters, allow_unused=True)
    squares = [
        gradient.detach().float().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    gradient_norm = float(torch.stack(squares).sum().sqrt().cpu()) if squares else 0.0
    nonzero_tensors = sum(
        int(gradient is not None and bool(gradient.detach().ne(0).any()))
        for gradient in gradients
    )
    report = {
        "status": "passed" if gradient_norm > 0.0 and nonzero_tensors > 0 else "failed",
        "route": adaptation.route,
        "checkpoint": str(args.checkpoint.resolve()),
        "sample_id": row["sample_id"],
        "caption": row["caption"],
        "question_dimensions": [question.dimension for question in u_batch.questions],
        "question_count": len(u_batch.questions),
        "candidate_mask": u_batch.candidate_mask.cpu().tolist(),
        "loss": float(loss.detach().cpu()),
        "adaptation_gradient_norm": gradient_norm,
        "adaptation_gradient_tensors_nonzero": nonzero_tensors,
        "adaptation_parameter_tensors": len(adaptation_parameters),
        "student_input_requires_grad": bool(predicted_u_latent.requires_grad),
        "strict_real_image_gate_valid": strict_details["gate_valid"],
        "strict_teacher_correct": strict_details.get("teacher_correct"),
        "student_correct": details["student_correct"],
        "decode_reencode_latent_mse": float(
            F.mse_loss(
                predicted_u_latent.float(), predicted_clean.detach().float()
            ).detach().cpu()
        ),
        "understanding_input": "predicted_image+question+choices_only",
        "generation_caption_excluded_from_u": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
