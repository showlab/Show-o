#!/usr/bin/env python3
"""Formal K=4 frozen-U TIFA-12 self-cycle continuation training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.optim import AdamW


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.utils import image_transform
from datasets.drsa_dataset import BprimeCocoBoxDataset
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
    prepare_cycle_questions,
    prepare_spatial_cycle_questions,
)
from evaluation.c6_mask2former_alignment.train_online_alignment import (
    CLASS_ALIASES,
    alignment_losses,
    class_conditioned_outputs,
    detector_raw,
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


FORMAT_VERSION = "showo2-tifa12-self-cycle-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", choices=("smoke", "train"), default="train")
    parser.add_argument("--max-steps", type=int, default=34012)
    parser.add_argument(
        "--save-steps",
        default="100,500,1000,2000,4000,6000,8000,10000,17000,25000,34012",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=4)
    parser.add_argument("--min-non-spatial-dimensions", type=int, default=2)
    parser.add_argument("--min-valid-questions", type=int, default=1)
    parser.add_argument("--gate-margin", type=float, default=0.0)
    parser.add_argument("--max-text-tokens", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument(
        "--generation-start",
        choices=("pure_noise", "reference_bridge"),
        default="pure_noise",
    )
    parser.add_argument("--allow-unpaired", action="store_true")
    parser.add_argument("--schedule-points", type=int, default=50)
    parser.add_argument("--time-shifting-factor", type=float, default=3.0)
    parser.add_argument("--t-min", type=float, default=0.10)
    parser.add_argument("--t-max", type=float, default=0.80)
    parser.add_argument("--status-every", type=int, default=1)
    parser.add_argument("--recovery-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--model", default="showlab/show-o2-1.5B")
    parser.add_argument("--vae-path", type=Path, default=ROOT / "Wan2.1_VAE.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replay-manifest", type=Path)
    parser.add_argument("--replay-shard-root", type=Path)
    parser.add_argument("--replay-probability", type=float, default=0.25)
    parser.add_argument("--replay-mask-weight", type=float, default=1.0)
    parser.add_argument("--replay-flow-weight", type=float, default=0.10)
    parser.add_argument("--teacher-consistency-weight", type=float, default=0.25)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument(
        "--detector-config",
        type=Path,
        default=ROOT.parents[1] / "mmdetection-2.28.2/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py",
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=ROOT.parents[1] / "models/geneval/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth",
    )
    parser.add_argument(
        "--object-names",
        type=Path,
        default=ROOT.parents[1] / "geneval/evaluation/object_names.txt",
    )
    return parser.parse_args()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_save_steps(value: str, max_steps: int) -> set[int]:
    steps = {int(item) for item in value.split(",") if item.strip()}
    if any(step < 1 or step > max_steps for step in steps):
        raise ValueError(f"save steps must be in [1,{max_steps}]: {sorted(steps)}")
    steps.add(max_steps)
    return steps


def step_seed(base_seed: int, sample_id: str, step: int) -> int:
    payload = f"{base_seed}\0{sample_id}\0{step}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def lr_scale(
    step: int,
    *,
    max_steps: int,
    warmup_steps: int,
    minimum_ratio: float,
) -> float:
    if step <= warmup_steps and warmup_steps > 0:
        return step / warmup_steps
    denominator = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / denominator))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def decode_prediction(vae: WanVAE, latents: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=vae.dtype):
        decoded = vae.model.decode(latents.unsqueeze(2), vae.scale).float()
    return decoded.squeeze(2).clamp(-1.0, 1.0)


def grad_norm(parameters: tuple[torch.nn.Parameter, ...]) -> tuple[float, int]:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    nonzero = sum(
        int(
            parameter.grad is not None
            and bool(parameter.grad.detach().ne(0).any())
        )
        for parameter in parameters
    )
    total = float(torch.stack(values).sum().sqrt().cpu()) if values else 0.0
    return total, nonzero


def save_checkpoint(
    path: Path,
    *,
    adaptation: Any,
    optimizer: AdamW,
    step: int,
    optimizer_updates: int,
    metrics: Mapping[str, Any],
    args: argparse.Namespace,
    init_sha256: str,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "adaptation_config": adaptation.config_dict(),
            "adaptation_state": adaptation.state_dict(),
            "format_version": FORMAT_VERSION,
            "init_checkpoint": str(args.init_checkpoint.resolve()),
            "init_checkpoint_sha256": init_sha256,
            "metrics": dict(metrics),
            "optimizer": optimizer.state_dict(),
            "optimizer_updates": optimizer_updates,
            "step": step,
            "training_config": {
                "gate_margin": args.gate_margin,
                "generation_start": args.generation_start,
                "allow_unpaired": args.allow_unpaired,
                "learning_rate": args.learning_rate,
                "manifest": str(args.manifest.resolve()),
                "max_questions": args.max_questions,
                "max_steps": args.max_steps,
                "min_valid_questions": args.min_valid_questions,
                "recovery_every": args.recovery_every,
                "rollout_steps": args.rollout_steps,
                "schedule_points": args.schedule_points,
                "seed": args.seed,
                "shard_root": str(args.shard_root.resolve()),
                "t_max": args.t_max,
                "t_min": args.t_min,
                "time_shifting_factor": args.time_shifting_factor,
                "replay_manifest": str(args.replay_manifest.resolve()) if args.replay_manifest else None,
                "replay_shard_root": str(args.replay_shard_root.resolve()) if args.replay_shard_root else None,
                "replay_probability": args.replay_probability,
                "replay_mask_weight": args.replay_mask_weight,
                "replay_flow_weight": args.replay_flow_weight,
                "teacher_consistency_weight": args.teacher_consistency_weight,
                "teacher_temperature": args.teacher_temperature,
            },
        },
        temporary,
    )
    temporary.replace(path)


def restore_metric_history(
    path: Path, step: int
) -> tuple[int, int, int, int]:
    if not path.is_file():
        return 0, 0, 0, 0
    retained: list[dict[str, Any]] = []
    valid_questions = 0
    questions = 0
    valid_steps = 0
    skipped_steps = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row.get("step", -1)) > step:
                continue
            retained.append(row)
            valid_questions += int(row["gate_valid_questions"])
            questions += int(row["question_count"])
            if bool(row["update_applied"]):
                valid_steps += 1
            else:
                skipped_steps += 1
    temporary = path.with_suffix(path.suffix + ".resume-tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)
    return valid_questions, questions, valid_steps, skipped_steps

def validate_args(args: argparse.Namespace, dataset_size: int) -> set[int]:
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("formal TIFA self-cycle training requires CUDA")
    if args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if args.mode == "train" and args.max_steps > dataset_size:
        raise ValueError(
            f"one-epoch protocol has {dataset_size} rows but max-steps={args.max_steps}"
        )
    if args.rollout_steps != 4:
        raise ValueError("formal protocol requires K=4")
    if args.max_questions != 4:
        raise ValueError("formal protocol requires four question slots")
    if args.min_valid_questions < 1:
        raise ValueError("min-valid-questions must be positive")
    if not 0.0 <= args.t_min < args.t_max <= 1.0:
        raise ValueError("expected 0 <= t-min < t-max <= 1")
    if not 0.0 <= args.minimum_lr_ratio <= 1.0:
        raise ValueError("minimum-lr-ratio must be in [0,1]")
    if args.learning_rate <= 0.0 or args.grad_clip <= 0.0:
        raise ValueError("learning-rate and grad-clip must be positive")
    if not 0.0 <= args.replay_probability <= 1.0:
        raise ValueError("replay-probability must be in [0,1]")
    if args.replay_manifest is not None and args.replay_shard_root is None:
        raise ValueError("replay-shard-root is required with replay-manifest")
    if args.replay_mask_weight < 0.0 or args.replay_flow_weight < 0.0:
        raise ValueError("replay weights must be non-negative")
    if args.teacher_consistency_weight < 0.0 or args.teacher_temperature <= 0.0:
        raise ValueError("teacher consistency weight must be non-negative and temperature positive")
    if args.recovery_every < 1:
        raise ValueError("recovery-every must be positive")
    return parse_save_steps(args.save_steps, args.max_steps)



def teacher_valid_mask(
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    gate_margin: float,
) -> torch.Tensor:
    mask = candidate_mask.to(dtype=torch.bool, device=teacher_logits.device)
    teacher = teacher_logits.detach().float().masked_fill(~mask, float("-inf"))
    targets = targets.to(device=teacher.device, dtype=torch.long)
    indices = torch.arange(targets.numel(), device=teacher.device)
    target_logits = teacher[indices, targets]
    competitors = teacher.masked_fill(
        F.one_hot(targets, num_classes=teacher.shape[-1]).bool(), float("-inf")
    ).amax(dim=-1)
    return teacher.argmax(dim=-1).eq(targets) & ((target_logits - competitors) > gate_margin)


def teacher_consistency_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    gate_margin: float,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    if temperature <= 0.0:
        raise ValueError("teacher temperature must be positive")
    valid = teacher_valid_mask(teacher_logits, targets, candidate_mask, gate_margin)
    mask = candidate_mask.to(dtype=torch.bool, device=student_logits.device)
    # A finite sentinel preserves zero probability for non-candidates without
    # creating the undefined 0 * (-inf - -inf) terms produced by KLDivLoss.
    sentinel = -1.0e4
    student = student_logits.float().masked_fill(~mask, sentinel)
    teacher = teacher_logits.detach().float().masked_fill(~mask, sentinel)
    t = float(temperature)
    student_log_probs = F.log_softmax(student / t, dim=-1)
    teacher_log_probs = F.log_softmax(teacher / t, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    per_class = teacher_probs * (teacher_log_probs - student_log_probs)
    per_class = per_class.masked_fill(~mask, 0.0)
    per_question = per_class.sum(dim=-1) * (t * t)
    count = int(valid.sum().detach().cpu())
    if count == 0:
        return student_logits.float().sum() * 0.0, 0
    return per_question[valid].mean(), count


def replay_mask_loss(
    detector: torch.nn.Module,
    model: Showo2Qwen2_5,
    vae: WanVAE,
    adaptation: Any,
    row: Mapping[str, Any],
    class_to_id: Mapping[str, int],
    transport: Any,
    tokenizer: Any,
    token_ids: Mapping[str, int],
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
    current_seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    phrases = [str(item["phrase"]) for item in row["objects"]]
    try:
        class_ids = [class_to_id[CLASS_ALIASES.get(phrase, phrase)] for phrase in phrases]
    except KeyError as error:
        raise KeyError(f"replay object class is missing from Mask2Former names: {phrases}") from error
    image = row["image"].unsqueeze(0).to(device=device, dtype=dtype)
    clean_latent = encode_images(vae, image, dtype)
    generator = torch.Generator(device=device).manual_seed(current_seed + 701)
    noise = torch.randn(clean_latent.shape, generator=generator, device=device, dtype=dtype)
    timestep = torch.empty((1,), device=device, dtype=dtype).uniform_(
        args.t_min, args.t_max, generator=generator
    )
    _, noisy_latent, velocity_target = transport.path_sampler.plan(timestep, noise, clean_latent)
    inputs = build_generation_inputs(
        str(row["prompt"]), tokenizer, token_ids, args.max_text_tokens,
        sequence_length, device, dtype,
    )
    def velocity_fn(state: torch.Tensor, current_timestep: torch.Tensor) -> torch.Tensor:
        velocity, _ = model_velocity(
            model, inputs, state, current_timestep, None, sequence_length, device
        )
        return velocity
    rollout = differentiable_euler_rollout(noisy_latent, timestep.float(), velocity_fn)
    predicted_clean = rollout.terminal_clean
    predicted_image = decode_prediction(vae, predicted_clean)
    with torch.no_grad():
        teacher_cls, teacher_masks = detector_raw(detector, image.float(), 800)
        teacher = class_conditioned_outputs(
            teacher_cls, teacher_masks, class_ids,
            query_temperature=0.25, foreground_temperature=0.25,
            quality_mode="official",
        )
        teacher = {key: value.detach() for key, value in teacher.items()}
    student_cls, student_masks = detector_raw(detector, predicted_image, 800)
    student = class_conditioned_outputs(
        student_cls, student_masks, class_ids,
        query_temperature=0.25, foreground_temperature=0.25,
        quality_mode="official",
    )
    losses, details = alignment_losses(
        student, teacher, [tuple(str(value) for value in row["relations"])],
        object_threshold=0.3, object_margin=0.05, object_teacher_ratio=0.8,
        teacher_validity_mask=True, object_loss_mode="squared",
    )
    mask_weights = {
        "mask_bce": 0.01,
        "mask_dice": 0.07,
        "presence": 0.16,
        "relation_distill": 0.50,
        "relation_pass": 1.00,
        "object_detect": 0.0,
    }
    mask_loss = torch.stack([losses[name].float() * weight for name, weight in mask_weights.items()]).sum()
    flow_loss = torch.stack([
        F.mse_loss(value.float(), velocity_target.float()) for value in rollout.velocities
    ]).mean()
    total = float(args.replay_mask_weight) * mask_loss + float(args.replay_flow_weight) * flow_loss
    return total, {
        "mask_loss": float(mask_loss.detach().cpu()),
        "flow_loss": float(flow_loss.detach().cpu()),
        "teacher_object_valid": bool(details["teacher_object_valid"][0]),
        "teacher_relation_valid": bool(details["teacher_relation_valid"][0]),
        "sample_id": str(row["sample_id"]),
        "relations": list(row["relations"]),
    }


def main() -> int:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "checkpoints").mkdir(exist_ok=True)
    status_path = args.run_dir / "status.json"
    metrics_path = args.run_dir / "metrics.jsonl"
    started = time.time()
    status: dict[str, Any] = {
        "status": "initializing",
        "label": args.label,
        "mode": args.mode,
        "step": 0,
        "total_steps": args.max_steps,
        "percent": 0.0,
        "started_at": now_iso(),
        "updated_at": now_iso(),
    }
    atomic_json(status_path, status)

    try:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        device = torch.device(args.device)
        dtype = torch.bfloat16

        dataset = TifaCocoCycleDataset(
            args.manifest,
            shard_root=args.shard_root,
            image_transform=lambda image: image_transform(image, resolution=432),
            allow_unpaired=args.allow_unpaired,
        )
        save_steps = validate_args(args, len(dataset))
        replay_dataset = None
        detector = None
        class_to_id = None
        if args.replay_manifest is not None:
            replay_dataset = BprimeCocoBoxDataset(
                args.replay_manifest,
                args.replay_shard_root,
                image_transform=lambda image: image_transform(image, resolution=432),
            )
            from mmdet.apis import init_detector
            detector = init_detector(
                str(args.detector_config), str(args.detector_checkpoint), device=str(args.device)
            )
            detector.eval().requires_grad_(False)
            class_names = [
                line.strip() for line in args.object_names.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            class_to_id = {name: index for index, name in enumerate(class_names)}
        init_sha = sha256(args.init_checkpoint)
        source_checkpoint = args.resume_checkpoint or args.init_checkpoint
        payload = torch.load(source_checkpoint, map_location="cpu", weights_only=True)

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
            **checkpoint_model_kwargs(payload),
        )
        model.to(device=device, dtype=dtype).requires_grad_(False).eval()
        adaptation = configure_from_checkpoint(model, payload)
        adaptation.requires_grad_(True)
        adaptation.train()
        adaptation.set_enabled(True)
        parameters = tuple(adaptation.parameters())
        active_ids = {id(parameter) for parameter in parameters}
        unexpected = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in active_ids
        ]
        if unexpected:
            raise AssertionError(f"unexpected trainable parameters: {unexpected[:8]}")

        vae = WanVAE(vae_pth=str(args.vae_path), dtype=dtype, device=device)
        vae.model.requires_grad_(False).eval()
        latent_template = encode_images(
            vae, torch.zeros((1, 3, 432, 432), device=device, dtype=dtype), dtype
        )
        optimizer = AdamW(
            parameters,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
        start_step = 0
        optimizer_updates = 0
        if args.resume_checkpoint is not None:
            if payload.get("format_version") != FORMAT_VERSION:
                raise ValueError("resume checkpoint is not a TIFA-12 continuation")
            optimizer.load_state_dict(payload["optimizer"])
            start_step = int(payload["step"])
            optimizer_updates = int(payload.get("optimizer_updates", 0))
            if start_step >= args.max_steps:
                raise ValueError("resume checkpoint already reached max-steps")

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

        status.update({
            "status": "running",
            "adaptation": adaptation.config_dict(),
            "dataset_rows": len(dataset),
            "generation_caption_in_u": False,
            "gate_policy": "paired_strict_unpaired_official_answer",
            "init_checkpoint": str(args.init_checkpoint.resolve()),
            "init_checkpoint_sha256": init_sha,
            "manifest": str(args.manifest.resolve()),
            "optimizer_updates": optimizer_updates,
            "resume_checkpoint": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None,
            "rollout_steps": args.rollout_steps,
            "save_steps": sorted(save_steps),
            "shard_root": str(args.shard_root.resolve()),
            "step": start_step,
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "understanding_input": "predicted_image+question+choices_only",
            "mixed_objective": {
                "replay_manifest": str(args.replay_manifest.resolve()) if args.replay_manifest else None,
                "replay_shard_root": str(args.replay_shard_root.resolve()) if args.replay_shard_root else None,
                "replay_probability": args.replay_probability,
                "replay_mask_weight": args.replay_mask_weight,
                "replay_flow_weight": args.replay_flow_weight,
                "teacher_consistency_weight": args.teacher_consistency_weight,
                "teacher_temperature": args.teacher_temperature,
                "teacher_consistency": "real-image frozen-U KL on teacher-valid COCO questions",
            },
            "updated_at": now_iso(),
        })
        atomic_json(status_path, status)
        initial_state = {
            name: value.detach().float().cpu().clone()
            for name, value in adaptation.state_dict().items()
        } if args.mode == "smoke" else None

        if start_step:
            (
                valid_questions_total,
                questions_total,
                valid_steps,
                skipped_steps,
            ) = restore_metric_history(metrics_path, start_step)
            if valid_steps != optimizer_updates:
                raise RuntimeError("optimizer update count differs from metrics history")
        else:
            valid_questions_total = 0
            questions_total = 0
            valid_steps = 0
            skipped_steps = 0
        last_metrics: dict[str, Any] = {}
        train_started = time.time()
        for step in range(start_step + 1, args.max_steps + 1):
            step_started = time.time()
            row_index = (args.start_index + step - 1) % len(dataset)
            row = dataset[row_index]
            non_spatial_dimensions = set(row["dimensions"]) - {"spatial"}
            if "spatial" in row["dimensions"] and non_spatial_dimensions:
                qa_pairs = prepare_spatial_cycle_questions(
                    row,
                    max_questions=args.max_questions,
                    min_non_spatial_dimensions=min(
                        args.min_non_spatial_dimensions, len(non_spatial_dimensions)
                    ),
                    base_seed=args.seed,
                    cycle_step=step,
                )
            else:
                qa_pairs = prepare_cycle_questions(
                    row,
                    max_questions=args.max_questions,
                    base_seed=args.seed,
                    cycle_step=step,
                )
            u_batch = build_tifa_question_batch(
                qa_pairs,
                tokenizer,
                token_ids,
                max_text_tokens=args.max_text_tokens,
                sequence_length=sequence_length,
                device=device,
            )
            generation_inputs = build_generation_inputs(
                str(row["caption"]),
                tokenizer,
                token_ids,
                args.max_text_tokens,
                sequence_length,
                device,
                dtype,
            )
            reference_image_available = bool(row.get("reference_image_available", False))
            clean_latent = None
            teacher_logits = None
            if reference_image_available:
                clean_latent = encode_images(vae, row["image"].unsqueeze(0), dtype)
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

            current_seed = step_seed(args.seed, str(row["sample_id"]), step)
            if args.generation_start == "reference_bridge" and clean_latent is not None:
                rollout_times = select_rollout_times(
                    time_grid,
                    args.rollout_steps,
                    args.t_min,
                    args.t_max,
                    generator=torch.Generator(device="cpu").manual_seed(current_seed + 41),
                )
                noise = torch.randn(
                    clean_latent.shape,
                    generator=torch.Generator(device=device).manual_seed(current_seed + 73),
                    device=device,
                    dtype=dtype,
                )
                _, noisy_latent, _ = transport.path_sampler.plan(
                    rollout_times[:1], noise, clean_latent
                )
            else:
                index_grid = torch.linspace(
                    0, time_grid.numel() - 2, args.rollout_steps, device=device
                ).round().long()
                rollout_times = time_grid.index_select(0, index_grid)
                noisy_latent = torch.randn(
                    latent_template.shape,
                    generator=torch.Generator(device=device).manual_seed(current_seed + 73),
                    device=device,
                    dtype=dtype,
                )
                noise = noisy_latent

            def velocity_fn(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
                velocity, _ = model_velocity(
                    model,
                    generation_inputs,
                    state,
                    timestep,
                    None,
                    sequence_length,
                    device,
                )
                return velocity

            optimizer.zero_grad(set_to_none=True)
            rollout = differentiable_euler_rollout(
                noisy_latent, rollout_times, velocity_fn
            )
            predicted_clean = rollout.terminal_clean
            predicted_image = decode_prediction(vae, predicted_clean)
            predicted_u_latent = deterministic_image_latents(
                vae, predicted_image, dtype
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
            if teacher_logits is None:
                masked_student = student_logits.float().masked_fill(
                    ~u_batch.candidate_mask, float("-inf")
                )
                per_question_ce = F.cross_entropy(
                    masked_student, u_batch.targets, reduction="none"
                )
                loss = per_question_ce.mean()
                gate = {
                    "gate_policy": "unpaired_official_answer",
                    "step_valid": True,
                    "valid_question_count": int(u_batch.targets.numel()),
                    "question_count": int(u_batch.targets.numel()),
                    "valid_question_fraction": 1.0,
                }
            else:
                loss, gate = per_question_tifa_cycle_loss(
                    student_logits,
                    teacher_logits,
                    u_batch.targets,
                    u_batch.candidate_mask,
                    gate_margin=args.gate_margin,
                    min_valid_questions=args.min_valid_questions,
                )
            teacher_kl = student_logits.float().sum() * 0.0
            teacher_kl_questions = 0
            if (
                teacher_logits is not None
                and args.teacher_consistency_weight > 0.0
            ):
                teacher_kl, teacher_kl_questions = teacher_consistency_kl(
                    student_logits,
                    teacher_logits,
                    u_batch.targets,
                    u_batch.candidate_mask,
                    args.gate_margin,
                    args.teacher_temperature,
                )
                loss = loss + args.teacher_consistency_weight * teacher_kl
            replay_loss = student_logits.float().sum() * 0.0
            replay_details = None
            replay_applied = False
            if (
                replay_dataset is not None
                and detector is not None
                and class_to_id is not None
                and args.replay_probability > 0.0
                and random.Random(step_seed(args.seed, str(row["sample_id"]), step) + 997).random()
                < args.replay_probability
            ):
                replay_row = replay_dataset[(step - 1) % len(replay_dataset)]
                replay_loss, replay_details = replay_mask_loss(
                    detector, model, vae, adaptation, replay_row, class_to_id,
                    transport, tokenizer, token_ids, sequence_length, device, dtype,
                    args, step_seed(args.seed, str(replay_row["sample_id"]), step),
                )
                loss = loss + replay_loss
                replay_applied = True
            update_applied = False
            raw_grad_norm = 0.0
            nonzero_gradient_tensors = 0
            clipped_grad_norm = 0.0
            learning_rate = args.learning_rate * lr_scale(
                step,
                max_steps=args.max_steps,
                warmup_steps=min(args.warmup_steps, args.max_steps),
                minimum_ratio=args.minimum_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            if gate["step_valid"]:
                if not bool(torch.isfinite(loss.detach()).item()):
                    def _finite_info(value):
                        value = value.detach().float()
                        safe = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                        return {
                            "finite": bool(torch.isfinite(value).all().item()),
                            "min": float(safe.min().cpu()),
                            "max": float(safe.max().cpu()),
                        }
                    raise FloatingPointError(
                        "non-finite mixed loss at step %d: loss=%s teacher_kl=%s replay_loss=%s"
                        % (step, _finite_info(loss), _finite_info(teacher_kl), _finite_info(replay_loss))
                    )
                loss.backward()
                raw_grad_norm, nonzero_gradient_tensors = grad_norm(parameters)
                clipped = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                clipped_grad_norm = float(clipped.detach().float().cpu())
                optimizer.step()
                optimizer_updates += 1
                valid_steps += 1
                update_applied = True
            else:
                skipped_steps += 1

            valid_questions = int(gate["valid_question_count"])
            valid_questions_total += valid_questions
            questions_total += int(gate["question_count"])
            elapsed = time.time() - train_started
            completed_this_process = step - start_step
            rate = completed_this_process / max(elapsed, 1e-9)
            eta = (args.max_steps - step) / max(rate, 1e-9)
            masked_student = student_logits.float().masked_fill(
                ~u_batch.candidate_mask, float("-inf")
            )
            last_metrics = {
                "decode_reencode_latent_mse": float(F.mse_loss(
                    predicted_u_latent.float(), predicted_clean.detach().float()
                ).detach().cpu()),
                "dimensions": [question.dimension for question in u_batch.questions],
                "elapsed_seconds": time.time() - started,
                "gate_valid_question_fraction": gate["valid_question_fraction"],
                "gate_valid_questions": valid_questions,
                "grad_norm_before_clip": raw_grad_norm,
                "grad_tensors_nonzero": nonzero_gradient_tensors,
                "learning_rate": learning_rate,
                "loss": float(loss.detach().cpu()),
                "teacher_consistency_kl": float(teacher_kl.detach().cpu()),
                "teacher_consistency_questions": teacher_kl_questions,
                "replay_applied": replay_applied,
                "replay_loss": float(replay_loss.detach().cpu()),
                "replay_details": replay_details,
                "optimizer_updates": optimizer_updates,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
                "question_count": gate["question_count"],
                "relation_count": int(row.get("relation_count", 0)),
                "rollout_times": [float(value) for value in rollout.times.detach().cpu()],
                "sample_id": row["sample_id"],
                "step": step,
                "step_seconds": time.time() - step_started,
                "student_correct": int(masked_student.argmax(dim=-1).eq(u_batch.targets).sum().cpu()),
                "update_applied": update_applied,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(last_metrics, sort_keys=True) + "\n")

            checkpoint = None
            if step in save_steps:
                checkpoint = args.run_dir / "checkpoints" / f"step-{step:06d}.pt"
            elif step % args.recovery_every == 0:
                checkpoint = args.run_dir / "checkpoints" / "recovery.pt"
            if checkpoint is not None:
                save_checkpoint(
                    checkpoint,
                    adaptation=adaptation,
                    optimizer=optimizer,
                    step=step,
                    optimizer_updates=optimizer_updates,
                    metrics=last_metrics,
                    args=args,
                    init_sha256=init_sha,
                )
                (args.run_dir / "latest_checkpoint.txt").write_text(str(checkpoint) + "\n")

            if step % args.status_every == 0 or step == args.max_steps:
                status.update({
                    "status": "running",
                    "step": step,
                    "percent": 100.0 * step / args.max_steps,
                    "optimizer_updates": optimizer_updates,
                    "valid_steps": valid_steps,
                    "skipped_steps": skipped_steps,
                    "valid_question_fraction": valid_questions_total / max(questions_total, 1),
                    "eta_seconds": eta,
                    "throughput_steps_per_second": rate,
                    "last_metrics": last_metrics,
                    "updated_at": now_iso(),
                })
                atomic_json(status_path, status)
            if step % args.log_every == 0 or step == args.max_steps:
                print(
                    f"step={step}/{args.max_steps} "
                    f"percent={100.0 * step / args.max_steps:.3f}% "
                    f"loss={last_metrics['loss']:.6f} "
                    f"gate={valid_questions}/{gate['question_count']} "
                    f"updates={optimizer_updates} eta_seconds={eta:.1f}",
                    flush=True,
                )
            optimizer.zero_grad(set_to_none=True)
            del (
                clean_latent,
                generation_inputs,
                loss,
                masked_student,
                noise,
                noisy_latent,
                predicted_clean,
                predicted_image,
                predicted_u_latent,
                rollout,
                student_logits,
                teacher_logits,
            )
            if step % 50 == 0:
                torch.cuda.empty_cache()
        smoke_checks: dict[str, Any] = {}
        if initial_state is not None:
            final_state = adaptation.state_dict()
            changed = sum(
                int(not torch.equal(initial_state[name], value.detach().float().cpu()))
                for name, value in final_state.items()
            )
            smoke_checks = {
                "adaptation_state_tensors_changed": changed,
                "optimizer_state_entries": len(optimizer.state),
                "optimizer_update_applied": optimizer_updates > 0,
                "only_selected_adaptation_trainable": not unexpected,
            }
            if not all((changed > 0, len(optimizer.state) > 0, optimizer_updates > 0)):
                raise RuntimeError(f"optimizer smoke failed: {smoke_checks}")

        status.update({
            "status": "completed",
            "step": args.max_steps,
            "percent": 100.0,
            "optimizer_updates": optimizer_updates,
            "valid_steps": valid_steps,
            "skipped_steps": skipped_steps,
            "valid_question_fraction": valid_questions_total / max(questions_total, 1),
            "eta_seconds": 0.0,
            "last_metrics": last_metrics,
            "mixed_objective": {
                "replay_manifest": str(args.replay_manifest.resolve()) if args.replay_manifest else None,
                "replay_probability": args.replay_probability,
                "replay_mask_weight": args.replay_mask_weight,
                "replay_flow_weight": args.replay_flow_weight,
                "teacher_consistency_weight": args.teacher_consistency_weight,
                "teacher_consistency_temperature": args.teacher_temperature,
            },
            "smoke_checks": smoke_checks,
            "updated_at": now_iso(),
        })
        atomic_json(status_path, status)
        (args.run_dir / "TRAINING_COMPLETED").touch()
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        status.update({
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "updated_at": now_iso(),
        })
        atomic_json(status_path, status)
        (args.run_dir / "TRAINING_FAILED").touch()
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
