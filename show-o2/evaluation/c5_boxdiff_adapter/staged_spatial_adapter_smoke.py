#!/usr/bin/env python3
"""End-to-end staged native + BoxDiff loss smoke for the DRSA adapter."""

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
import torch.nn.functional as F
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
from evaluation.c5_boxdiff_adapter.boxdiff_adapter_smoke import (
    g_box_energy,
    phrase_embeddings,
    scalarize_details,
)
from evaluation.c5_boxdiff_adapter.native_objective_warmup_smoke import (
    optimizer_group_update_stats,
    snapshot_optimizer_groups,
)
from evaluation.c5_boxdiff_adapter.staged_adapter_loss import (
    StagedAdapterLossConfig,
    compose_staged_loss,
)
from guidance.energy import EnergyConfig, LayoutEnergy
from guidance.layout import LayoutSpec, map_layout_tokens, parse_position_layout
from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
from models.misc import get_text_tokenizer
from transport import create_transport
from utils import path_to_llm_name


TASKS = {"spright_t2i", "vsr_mmu"}
RELATION_TEXT = {
    "left": "left of",
    "right": "right of",
    "above": "above",
    "below": "below",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-mode", choices=("smoke", "train"), default="smoke")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/autodl-tmp/data/drsa"),
    )
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--model", default="showlab/show-o2-1.5B")
    parser.add_argument("--vae-path", type=Path, default=ROOT / "Wan2.1_VAE.pth")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--native-steps", type=int, default=1)
    parser.add_argument("--spatial-ramp-steps", type=int, default=2)
    parser.add_argument(
        "--task-pattern",
        default="spright_t2i,spright_t2i,vsr_mmu,spright_t2i",
    )
    parser.add_argument("--save-every", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument("--max-spatial-skips", type=int, default=64)
    parser.add_argument("--lr-u", type=float, default=1e-5)
    parser.add_argument("--lr-g", type=float, default=5e-6)
    parser.add_argument("--lr-relation", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--map-temperature", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_task_pattern(value: str) -> tuple[str, ...]:
    tasks = tuple(item.strip() for item in value.split(",") if item.strip())
    if not tasks:
        raise ValueError("task-pattern must contain at least one task")
    unknown = sorted(set(tasks) - TASKS)
    if unknown:
        raise ValueError(f"unsupported task-pattern entries: {unknown}")
    return tasks


def build_generation_inputs(
    caption: str,
    tokenizer: Any,
    token_ids: dict[str, int],
    max_text_tokens: int,
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    sequence = build_generation_sequence(
        caption,
        tokenizer,
        token_ids,
        max_text_tokens,
        sequence_length,
    )
    text_tokens, _labels, positions, text_mask, image_mask = [
        value.unsqueeze(0).to(device) for value in sequence
    ]
    attention_mask = omni_attn_mask_naive(
        1,
        sequence_length,
        positions,
        device,
    ).to(dtype)
    return {
        "text_tokens": text_tokens,
        "modality_positions": positions,
        "text_masks": text_mask,
        "image_masks": image_mask,
        "attention_mask": attention_mask,
    }


def canonical_spatial_layout(
    spatial_caption: str,
    tokenizer: Any,
    max_text_tokens: int,
) -> tuple[str, LayoutSpec]:
    parsed = parse_position_layout(spatial_caption)
    if any(len(item.phrase.split()) > 14 for item in parsed.objects):
        raise ValueError("parsed object phrase is too long for reliable pseudo grounding")
    relation = RELATION_TEXT[parsed.relation]
    guided_caption = (
        f"a photo of {parsed.objects[0].phrase} {relation} "
        f"{parsed.objects[1].phrase}"
    )
    layout = map_layout_tokens(parse_position_layout(guided_caption), tokenizer)
    largest_index = max(
        index
        for item in layout.objects
        for index in item.token_indices
    )
    if largest_index >= max_text_tokens + 1:
        raise ValueError("guided object tokens would be truncated")
    return guided_caption, layout


def model_velocity(
    model: Showo2Qwen2_5,
    inputs: dict[str, torch.Tensor],
    noisy_latents: torch.Tensor,
    timestep: torch.Tensor,
    relation_codes: torch.Tensor | None,
    sequence_length: int,
    device: torch.device,
    *,
    output_attentions: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor | None, ...] | None]:
    output = model(
        text_tokens=inputs["text_tokens"],
        image_latents=noisy_latents,
        t=timestep,
        attention_mask=inputs["attention_mask"],
        text_masks=inputs["text_masks"],
        image_masks=inputs["image_masks"],
        modality_positions=inputs["modality_positions"],
        output_attentions=output_attentions,
        output_hidden_states=True,
        max_seq_len=sequence_length,
        device=device,
        drsa_relation_codes=relation_codes,
    )
    if output_attentions:
        _logits, velocity, attentions = output
        return velocity, tuple(attentions)
    _logits, velocity = output
    return velocity, None


def next_spright_batch(
    iterator: Any,
    loader: DataLoader,
) -> tuple[dict[str, Any], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def prepare_spatial_batch(
    iterator: Any,
    loader: DataLoader,
    tokenizer: Any,
    max_text_tokens: int,
    max_skips: int,
) -> tuple[dict[str, Any], Any, str, LayoutSpec, int]:
    for skipped in range(max_skips + 1):
        batch, iterator = next_spright_batch(iterator, loader)
        spatial_caption = str(batch["spatial_caption"][0])
        try:
            guided_caption, layout = canonical_spatial_layout(
                spatial_caption,
                tokenizer,
                max_text_tokens,
            )
            return batch, iterator, guided_caption, layout, skipped
        except (KeyError, TypeError, ValueError):
            continue
    raise RuntimeError(
        f"no parseable left/right/above/below SPRIGHT sample in {max_skips + 1} rows"
    )


def load_adapter_checkpoint(
    model: Showo2Qwen2_5,
    checkpoint_path: Path | None,
) -> int:
    if checkpoint_path is None:
        return 0
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = payload.get("drsa_adapter")
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint has no drsa_adapter state: {checkpoint_path}")
    if any(key.startswith("generation_spatial.") for key in state):
        model.drsa_adapter.enable_generation_spatial()
    model.drsa_adapter.load_state_dict(state, strict=True)
    return int(payload.get("step", 0))


def main() -> int:
    args = parse_args()
    if args.max_steps < 1 or args.save_every < 1:
        raise ValueError("max-steps and save-every must be positive")
    if args.max_spatial_skips < 0:
        raise ValueError("max-spatial-skips must be non-negative")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("staged spatial loss smoke requires CUDA")
    if not args.data_manifest.is_file():
        raise FileNotFoundError(args.data_manifest)
    task_pattern = parse_task_pattern(args.task_pattern)
    schedule = StagedAdapterLossConfig(
        native_steps=args.native_steps,
        ramp_steps=args.spatial_ramp_steps,
    )
    if (
        args.run_mode == "smoke"
        and schedule.weights_at(args.max_steps).stage != "joint_stabilize"
    ):
        raise ValueError(
            "smoke must cross native_anchor, spatial_ramp, and joint_stabilize"
        )

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
    init_step = load_adapter_checkpoint(model, args.init_checkpoint)
    model.eval()
    model.drsa_adapter.requires_grad_(True)
    model.drsa_adapter.train()
    vae = WanVAE(
        vae_pth=str(args.vae_path),
        dtype=dtype,
        device=device,
    )

    optimizer = AdamW(
        [
            {
                "params": model.drsa_adapter.understanding.parameters(),
                "lr": args.lr_u,
                "name": "u",
            },
            {
                "params": model.drsa_adapter.generation.parameters(),
                "lr": args.lr_g,
                "name": "g",
            },
            {
                "params": model.drsa_adapter.relation_encoder.parameters(),
                "lr": args.lr_relation,
                "name": "relation",
            },
        ],
        weight_decay=args.weight_decay,
    )
    warmup = max(1, min(args.warmup_steps, args.max_steps))

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
    spright_loader = DataLoader(
        SprightCocoIterableDataset(
            args.data_root / "spright_coco" / "data",
            image_transform=transform,
            shuffle_shards=False,
            shuffle_buffer=1,
            seed=args.seed,
        ),
        batch_size=1,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    vsr_loader = DataLoader(
        VsrRelationDataset(
            args.data_root / "vsr" / "annotations" / "zeroshot" / "train.jsonl",
            args.data_root / "vsr" / "vsr_images.zip",
            image_transform=transform,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    spright_iterator = iter(spright_loader)
    vsr_iterator = iter(vsr_loader)

    energy = LayoutEnergy(
        EnergyConfig(
            layers=(-1,),
            inside_weight=0.0,
            cross_weight=0.0,
            relation_weight=1.0,
            relation_margin=0.20,
            boxdiff_inner_weight=1.0,
            boxdiff_outer_weight=0.5,
            boxdiff_corner_weight=0.25,
            boxdiff_p=0.20,
            boxdiff_corner_radius=1,
            boxdiff_text_scale=100.0,
        )
    )
    hook_state: dict[str, torch.Tensor] = {}

    def generation_hook(_module: Any, _inputs: Any, output: Any) -> None:
        hook_state["hidden"] = output[0] if isinstance(output, tuple) else output

    hook_handle = model.drsa_adapter.generation.register_forward_hook(
        generation_hook
    )
    counts = {
        "spright_t2i": 0,
        "vsr_mmu": 0,
        "spatial_samples": 0,
        "spatial_skips": 0,
    }
    records: list[dict[str, Any]] = []
    status.update(
        {
            "status": "running",
            "phase": "ready",
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "base_model_frozen": all(
                not parameter.requires_grad
                for name, parameter in model.named_parameters()
                if not name.startswith("drsa_adapter.")
            ),
            "init_checkpoint": (
                str(args.init_checkpoint.resolve())
                if args.init_checkpoint is not None
                else None
            ),
            "init_checkpoint_step": init_step,
            "task_pattern": list(task_pattern),
            "missing_adapter_keys": sum(
                key.startswith("drsa_adapter.")
                for key in loading_info.get("missing_keys", [])
            ),
            "updated_at": now_iso(),
        }
    )
    atomic_json(status_path, status)

    try:
        for step in range(1, args.max_steps + 1):
            task = task_pattern[(step - 1) % len(task_pattern)]
            weights = schedule.weights_at(step)
            spatial_active = (
                task == "spright_t2i"
                and (weights.box_u > 0.0 or weights.box_g > 0.0)
            )
            optimizer.zero_grad(set_to_none=True)
            raw_losses: dict[str, torch.Tensor | None] = {}
            task_metrics: dict[str, Any]

            if task == "spright_t2i":
                if spatial_active:
                    (
                        batch,
                        spright_iterator,
                        guided_caption,
                        layout,
                        skipped,
                    ) = prepare_spatial_batch(
                        spright_iterator,
                        spright_loader,
                        tokenizer,
                        args.max_text_tokens,
                        args.max_spatial_skips,
                    )
                    counts["spatial_samples"] += 1
                    counts["spatial_skips"] += skipped
                else:
                    batch, spright_iterator = next_spright_batch(
                        spright_iterator,
                        spright_loader,
                    )
                    guided_caption = None
                    layout = None
                    skipped = 0

                images = batch["image"]
                original_caption = str(batch["original_caption"][0])
                spatial_caption = str(batch["spatial_caption"][0])
                clean_latents = encode_images(vae, images, dtype)
                with torch.no_grad():
                    t, x0, x1 = transport.sample(clean_latents)
                    t, noisy_latents, velocity_target = (
                        transport.path_sampler.plan(t, x0, x1)
                    )
                    timestep = t.to(device=device, dtype=dtype)
                    noisy_latents = noisy_latents.to(dtype)
                    velocity_target = velocity_target.to(dtype)

                native_inputs = build_generation_inputs(
                    original_caption,
                    tokenizer,
                    token_ids,
                    args.max_text_tokens,
                    sequence_length,
                    device,
                    dtype,
                )
                native_code = relation_code(
                    model,
                    tokenizer,
                    [original_caption],
                    device,
                )

                if not spatial_active:
                    _logits, native_flow = model(
                        **native_inputs,
                        image_latents=noisy_latents,
                        t=timestep,
                        image_labels=velocity_target,
                        output_hidden_states=True,
                        max_seq_len=sequence_length,
                        device=device,
                        drsa_relation_codes=native_code,
                    )
                    raw_losses["native_flow"] = native_flow.float()
                    task_metrics = {
                        "original_caption": original_caption,
                        "spatial_caption": spatial_caption,
                        "loss_native_flow": float(native_flow.detach()),
                        "spatial_active": False,
                    }
                else:
                    native_velocity, _ = model_velocity(
                        model,
                        native_inputs,
                        noisy_latents,
                        timestep,
                        native_code,
                        sequence_length,
                        device,
                    )
                    with torch.no_grad():
                        teacher_velocity, _ = model_velocity(
                            model,
                            native_inputs,
                            noisy_latents,
                            timestep,
                            None,
                            sequence_length,
                            device,
                        )

                    assert guided_caption is not None and layout is not None
                    guided_inputs = build_generation_inputs(
                        guided_caption,
                        tokenizer,
                        token_ids,
                        args.max_text_tokens,
                        sequence_length,
                        device,
                        dtype,
                    )
                    relation_prompt = (
                        f"{layout.objects[0].phrase} {layout.relation} "
                        f"{layout.objects[1].phrase}"
                    )
                    guided_code = relation_code(
                        model,
                        tokenizer,
                        [relation_prompt],
                        device,
                    )
                    phrases = phrase_embeddings(
                        model,
                        tokenizer,
                        layout,
                        device,
                    )
                    hook_state.clear()
                    guided_velocity, attentions = model_velocity(
                        model,
                        guided_inputs,
                        noisy_latents,
                        timestep,
                        guided_code,
                        sequence_length,
                        device,
                        output_attentions=True,
                    )
                    if not attentions or all(item is None for item in attentions):
                        raise RuntimeError(
                            "guided forward did not return Qwen attentions"
                        )
                    if "hidden" not in hook_state:
                        raise RuntimeError(
                            "guided forward did not capture G adapter hidden states"
                        )

                    u_total, _u_per_sample, u_details = energy.evaluate(
                        attentions,
                        [layout],
                        guided_inputs["modality_positions"],
                        (27, 27),
                        bool(model.config.add_time_embeds),
                    )
                    u_box = torch.stack(
                        [item["boxdiff_energy"] for item in u_details]
                    ).float().mean()
                    u_relation = torch.stack(
                        [
                            item.get(
                                "relation_energy",
                                u_total.new_zeros(()),
                            )
                            for item in u_details
                        ]
                    ).float().mean()
                    g_box, g_relation, g_details = g_box_energy(
                        hook_state["hidden"],
                        layout,
                        phrases,
                        height=27,
                        width=27,
                        temperature=args.map_temperature,
                    )

                    native_flow = F.mse_loss(
                        native_velocity.float(),
                        velocity_target.float(),
                    )
                    guided_flow = F.mse_loss(
                        guided_velocity.float(),
                        velocity_target.float(),
                    )
                    retention = F.mse_loss(
                        native_velocity.float(),
                        teacher_velocity.detach().float(),
                    )
                    raw_losses.update(
                        {
                            "native_flow": native_flow,
                            "guided_flow": guided_flow,
                            "retention": retention,
                            "box_u": u_box,
                            "box_g": g_box,
                            "relation_u": u_relation,
                            "relation_g": g_relation,
                        }
                    )
                    task_metrics = {
                        "original_caption": original_caption,
                        "source_spatial_caption": spatial_caption,
                        "guided_caption": guided_caption,
                        "layout_relation": layout.relation,
                        "layout_objects": [
                            item.phrase for item in layout.objects
                        ],
                        "spatial_active": True,
                        "spatial_skips_before_sample": skipped,
                        "loss_native_flow": float(native_flow.detach()),
                        "loss_guided_flow": float(guided_flow.detach()),
                        "loss_retention": float(retention.detach()),
                        "loss_box_u": float(u_box.detach()),
                        "loss_box_g": float(g_box.detach()),
                        "loss_relation_u": float(u_relation.detach()),
                        "loss_relation_g": float(g_relation.detach()),
                        "u_details": scalarize_details(u_details),
                        "g_details": g_details,
                    }
                counts["spright_t2i"] += 1
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
                    row,
                    tokenizer,
                    token_ids,
                    args.max_text_tokens,
                    sequence_length,
                )
                text_tokens, text_labels, positions, text_mask, image_mask = [
                    value.unsqueeze(0).to(device) for value in sequence
                ]
                attention_mask = omni_attn_mask_naive(
                    1,
                    sequence_length,
                    positions,
                    device,
                ).to(dtype)
                code = relation_code(
                    model,
                    tokenizer,
                    [str(row["relation"])],
                    device,
                )
                _logits, loss_ntp = model.forward_und_only(
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
                raw_losses["mmu"] = loss_ntp.float()
                counts["vsr_mmu"] += 1
                task_metrics = {
                    "caption": str(row["caption"]),
                    "relation": str(row["relation"]),
                    "label": int(row["label"]),
                    "loss_mmu": float(loss_ntp.detach()),
                    "spatial_active": False,
                }

            loss, weighted_components = compose_staged_loss(
                raw_losses,
                weights,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite staged loss at step {step}: {float(loss)}"
                )
            loss.backward()
            grad_u = adapter_grad_norm(model.drsa_adapter.understanding)
            grad_g = adapter_grad_norm(model.drsa_adapter.generation)
            grad_relation = adapter_grad_norm(
                model.drsa_adapter.relation_encoder
            )
            if task == "vsr_mmu" and grad_g != 0.0:
                raise AssertionError(f"MMU step produced G gradient: {grad_g}")
            if spatial_active and (grad_u == 0.0 or grad_g == 0.0):
                raise AssertionError(
                    f"spatial step did not reach both routes: U={grad_u}, G={grad_g}"
                )

            global_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.drsa_adapter.parameters(),
                    args.grad_clip,
                ).detach().cpu()
            )
            clipped_grad_u = adapter_grad_norm(
                model.drsa_adapter.understanding
            )
            clipped_grad_g = adapter_grad_norm(
                model.drsa_adapter.generation
            )
            clipped_grad_relation = adapter_grad_norm(
                model.drsa_adapter.relation_encoder
            )
            applied_lrs = {
                str(group["name"]): float(group["lr"])
                for group in optimizer.param_groups
            }
            parameter_snapshots = snapshot_optimizer_groups(optimizer)
            optimizer.step()
            update_stats = optimizer_group_update_stats(
                optimizer,
                parameter_snapshots,
            )
            scheduler.step()
            if task == "vsr_mmu" and update_stats["g"]["update_norm"] != 0.0:
                raise AssertionError(
                    f"MMU step updated G parameters: {update_stats['g']['update_norm']}"
                )

            elapsed = time.time() - started
            record = {
                "step": step,
                "stage": weights.as_dict(),
                "task": task,
                "loss": float(loss.detach()),
                "raw_losses": {
                    name: float(value.detach())
                    for name, value in raw_losses.items()
                    if value is not None
                },
                "weighted_components": {
                    name: float(value.detach())
                    for name, value in weighted_components.items()
                },
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
                "lr_u": applied_lrs["u"],
                "lr_g": applied_lrs["g"],
                "lr_relation": applied_lrs["relation"],
                "counts": dict(counts),
                "elapsed_seconds": elapsed,
                **task_metrics,
            }
            records.append(record)
            status.update(
                {
                    "status": "running",
                    "phase": f"{weights.stage}:{task}",
                    "step": step,
                    "total_steps": args.max_steps,
                    "percent": 100.0 * step / args.max_steps,
                    "counts": counts,
                    "metrics": record,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed / step * (args.max_steps - step),
                    "updated_at": now_iso(),
                }
            )
            atomic_json(status_path, status)
            atomic_json(
                args.run_dir / "staged_loss_metrics.json",
                {"records": records},
            )
            print(
                f"step={step}/{args.max_steps} stage={weights.stage} "
                f"task={task} loss={record['loss']:.6f} "
                f"u_grad={grad_u:.4e} g_grad={grad_g:.4e} "
                f"rel_grad={grad_relation:.4e} "
                f"u_update={record['update_u']:.4e} "
                f"g_update={record['update_g']:.4e}",
                flush=True,
            )
            if step % args.save_every == 0 or step == args.max_steps:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    step,
                    args.run_dir,
                    record,
                )

        if args.run_mode == "smoke":
            stages_seen = {record["stage"]["stage"] for record in records}
            required_stages = {
                "native_anchor",
                "spatial_ramp",
                "joint_stabilize",
            }
            if stages_seen != required_stages:
                raise AssertionError(
                    f"stage traversal incomplete: seen={sorted(stages_seen)}"
                )
            spatial_records = [
                record for record in records if record.get("spatial_active")
            ]
            if len(spatial_records) < 2:
                raise AssertionError("smoke did not run both ramp and joint spatial steps")

        status.update(
            {
                "status": "completed",
                "phase": "completed",
                "percent": 100.0,
                "updated_at": now_iso(),
                "elapsed_seconds": time.time() - started,
            }
        )
        atomic_json(status_path, status)
        atomic_json(
            args.run_dir / "metadata.json",
            {
                "status": "completed",
                "protocol": (
                    "staged-native-boxdiff-smoke-v1"
                    if args.run_mode == "smoke"
                    else "staged-native-boxdiff-train-v1"
                ),
                "run_mode": args.run_mode,
                "base_model_frozen": True,
                "init_checkpoint": (
                    str(args.init_checkpoint.resolve())
                    if args.init_checkpoint is not None
                    else None
                ),
                "init_checkpoint_step": init_step,
                "task_pattern": list(task_pattern),
                "loss_schedule": {
                    "native_steps": args.native_steps,
                    "spatial_ramp_steps": args.spatial_ramp_steps,
                    "joint_weights": schedule.weights_at(args.max_steps).as_dict(),
                },
                "learning_rates": {
                    "u": args.lr_u,
                    "g": args.lr_g,
                    "relation": args.lr_relation,
                },
                "teacher": "same frozen model with drsa_relation_codes=None",
                "spatial_source": "SPRIGHT caption canonicalized to rule pseudo boxes",
            },
        )
        return 0
    except BaseException as error:
        status.update(
            {
                "status": "failed",
                "phase": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "updated_at": now_iso(),
            }
        )
        atomic_json(status_path, status)
        raise
    finally:
        hook_handle.remove()


if __name__ == "__main__":
    raise SystemExit(main())
