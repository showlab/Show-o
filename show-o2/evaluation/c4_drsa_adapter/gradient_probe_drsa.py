#!/usr/bin/env python3
"""Measure DRSA U/G gradients on fixed SPRIGHT and VSR examples.

This probe deliberately evaluates the losses that are already used by the
training script.  It does not optimize parameters and it does not introduce
an alignment loss.  U (understanding) and G (generation) are disjoint
parameter spaces, so their gradients are compared by scale and update size;
cosines are only reported for the shared relation encoder or for the same
route across two tasks.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--showo-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/showo2_1.5b_demo_432x432.yaml")
    parser.add_argument("--model", default="showlab/show-o2-1.5B")
    parser.add_argument("--vae-path", type=Path, default=ROOT / "Wan2.1_VAE.pth")
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/data/drsa"))
    parser.add_argument("--checkpoint-3k", type=Path,
                        default=Path("/root/autodl-tmp/runs/showo2_drsa_train/drsa-train-20260806T165335Z/checkpoints/step-003000.pt"))
    parser.add_argument("--checkpoint-20k", type=Path,
                        default=Path("/root/autodl-tmp/runs/showo2_drsa_train/drsa-train-20k-20260807T065000Z/checkpoints/step-020000.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--spright-samples", type=int, default=4)
    parser.add_argument("--vsr-samples", type=int, default=4)
    parser.add_argument("--spright-skip", type=int, default=0)
    parser.add_argument("--vsr-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-type", choices=("bfloat16", "float32"), default="bfloat16")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def dtype_from_name(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def build_generation_sequence(caption, tokenizer, token_ids, max_text_tokens, sequence_length):
    from datasets.utils import format_sequence_gen_qwen2_5

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


def encode_images(vae, images: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    with torch.no_grad():
        videos = images.to(device=device, dtype=dtype).unsqueeze(2)
        return vae.sample(videos, deterministic=True).squeeze(2).to(dtype)


def relation_code(model, tokenizer, texts: Iterable[str], device: torch.device) -> torch.Tensor:
    encoded = tokenizer(
        list(texts), add_special_tokens=False, padding=True, truncation=True,
        max_length=32, return_tensors="pt",
    )
    token_ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        features = model.showo.model.embed_tokens(token_ids)
    return model.drsa_adapter.encode_relation(features.detach(), mask)


def module_groups(adapter) -> dict[str, list[torch.nn.Parameter]]:
    return {
        "understanding": list(adapter.understanding.parameters()),
        "generation": list(adapter.generation.parameters()),
        "relation_encoder": list(adapter.relation_encoder.parameters()),
        "adapter": list(adapter.parameters()),
    }


def tensor_stats(values: list[torch.Tensor], parameter_values: list[torch.Tensor] | None = None) -> dict[str, Any]:
    values = [value.detach().float().reshape(-1) for value in values if value is not None]
    if not values:
        return {"norm": 0.0, "rms": 0.0, "max_abs": 0.0, "nonzero": 0, "elements": 0}
    flat = torch.cat(values)
    result = {
        "norm": float(flat.norm().cpu()),
        "rms": float(flat.square().mean().sqrt().cpu()),
        "max_abs": float(flat.abs().max().cpu()),
        "nonzero": int((flat != 0).sum().cpu()),
        "elements": int(flat.numel()),
    }
    if parameter_values is not None:
        params = torch.cat([value.detach().float().reshape(-1) for value in parameter_values])
        result["parameter_rms"] = float(params.square().mean().sqrt().cpu())
        result["parameter_max_abs"] = float(params.abs().max().cpu())
    return result


def cosine(a: list[torch.Tensor], b: list[torch.Tensor]) -> float | None:
    aa = [x.detach().float().reshape(-1) for x in a if x is not None]
    bb = [x.detach().float().reshape(-1) for x in b if x is not None]
    if len(aa) != len(bb) or not aa:
        return None
    va, vb = torch.cat(aa), torch.cat(bb)
    denom = float(va.norm() * vb.norm())
    return float(torch.dot(va, vb).cpu() / denom) if denom else None


def adamw_updates(
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    gradients: dict[int, torch.Tensor],
    clip_scale: float,
) -> dict[int, torch.Tensor]:
    """Predict one AdamW update without mutating parameters or optimizer state."""
    updates: dict[int, torch.Tensor] = {}
    group = optimizer.param_groups[0]
    lr = float(group["lr"])
    beta1, beta2 = group["betas"]
    eps = float(group["eps"])
    weight_decay = float(group.get("weight_decay", 0.0))
    for index, parameter in enumerate(parameters):
        gradient = gradients.get(index)
        if gradient is None:
            continue
        grad = gradient.float() * clip_scale
        state = optimizer.state.get(parameter, {})
        old_step = state.get("step", 0.0)
        if torch.is_tensor(old_step):
            old_step = float(old_step.item())
        step = old_step + 1.0
        old_avg = state.get("exp_avg")
        old_sq = state.get("exp_avg_sq")
        if old_avg is None:
            old_avg = torch.zeros_like(parameter, dtype=torch.float32)
        else:
            old_avg = old_avg.detach().float()
        if old_sq is None:
            old_sq = torch.zeros_like(parameter, dtype=torch.float32)
        else:
            old_sq = old_sq.detach().float()
        avg = old_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
        sq = old_sq.mul(beta2).addcmul(grad, grad, value=1.0 - beta2)
        bias1 = 1.0 - beta1 ** step
        bias2 = 1.0 - beta2 ** step
        denom = sq.sqrt() / math.sqrt(bias2)
        denom.add_(eps)
        delta = (lr / bias1) * avg / denom
        if weight_decay:
            delta = delta + lr * weight_decay * parameter.detach().float()
        updates[index] = -delta
    return updates


def state_metrics(
    optimizer: torch.optim.Optimizer,
    groups: dict[str, list[torch.nn.Parameter]],
    parameters: list[torch.nn.Parameter],
    raw_gradients: dict[int, torch.Tensor],
    clip_scale: float,
) -> dict[str, Any]:
    index_by_id = {id(parameter): index for index, parameter in enumerate(parameters)}
    updates = adamw_updates(optimizer, parameters, raw_gradients, clip_scale)
    output: dict[str, Any] = {}
    for name, module_parameters in groups.items():
        indices = [index_by_id[id(parameter)] for parameter in module_parameters]
        raw = [raw_gradients.get(index) for index in indices]
        clipped = [gradient * clip_scale if gradient is not None else None for gradient in raw]
        update = [updates.get(index) for index in indices]
        values = [parameters[index].detach() for index in indices]
        output[name] = {
            "parameter_count": int(sum(parameters[index].numel() for index in indices)),
            "raw": tensor_stats(raw, values),
            "clipped": tensor_stats(clipped),
            "adamw_update": tensor_stats(update),
        }
    output["global"] = {
        "raw_norm": float(math.sqrt(sum(g.float().square().sum().item() for g in raw_gradients.values()))),
        "clip_scale": float(clip_scale),
        "adamw_lr": float(optimizer.param_groups[0]["lr"]),
    }
    return output


def evaluate_flow(model, batch: dict[str, Any], tokenizer, token_ids, transport, device, dtype, max_text_tokens, sequence_length):
    from datasets.utils import image_transform
    from models import omni_attn_mask_naive

    clean_latent = batch["clean_latent"]
    t = batch["t"]
    noisy_latent = batch["noisy_latent"]
    velocity_target = batch["velocity_target"]
    sequence = build_generation_sequence(batch["spatial_caption"], tokenizer, token_ids, max_text_tokens, sequence_length)
    text_tokens, _, modality_positions, text_mask, image_mask = [value.unsqueeze(0).to(device) for value in sequence]
    attention_mask = omni_attn_mask_naive(1, sequence_length, modality_positions, device).to(dtype)
    code = relation_code(model, tokenizer, [batch["spatial_caption"]], device)
    _, loss_flow = model(
        text_tokens=text_tokens,
        image_latents=noisy_latent,
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
    return loss_flow


def evaluate_vsr(model, batch: dict[str, Any], tokenizer, token_ids, device, dtype, max_text_tokens, sequence_length):
    from models import omni_attn_mask_naive

    sequence = build_understanding_sequence(batch, tokenizer, token_ids, max_text_tokens, sequence_length)
    text_tokens, text_labels, modality_positions, text_mask, image_mask = [value.unsqueeze(0).to(device) for value in sequence]
    attention_mask = omni_attn_mask_naive(1, sequence_length, modality_positions, device).to(dtype)
    code = relation_code(model, tokenizer, [str(batch["relation"])], device)
    t = torch.ones((1,), device=device, dtype=dtype)
    _, loss_ntp = model.forward_und_only(
        text_tokens=text_tokens,
        image_latents=batch["clean_latent"],
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
    return loss_ntp


def capture_gradients(model, loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> dict[int, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    loss.backward()
    return {
        index: parameter.grad.detach().float().clone()
        for index, parameter in enumerate(parameters)
        if parameter.grad is not None
    }


def make_optimizer(model, checkpoint_payload: dict[str, Any] | None, args) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(
        model.drsa_adapter.parameters(), lr=args.learning_rate,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay,
    )
    if checkpoint_payload is not None and "optimizer" in checkpoint_payload:
        optimizer.load_state_dict(checkpoint_payload["optimizer"])
    return optimizer


def load_checkpoint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def prepare_fixed_batches(args, tokenizer, token_ids, config, vae, transport, device, dtype):
    from datasets.drsa_dataset import SprightCocoIterableDataset, VsrRelationDataset
    from datasets.utils import image_transform

    transform = partial(image_transform, resolution=int(config.dataset.preprocessing.resolution))
    spright = SprightCocoIterableDataset(
        args.data_root / "spright_coco" / "data", image_transform=transform,
        shuffle_shards=False, shuffle_buffer=0, seed=args.seed,
    )
    spright_iter = iter(spright)
    spright_rows = []
    for _ in range(args.spright_skip + args.spright_samples):
        spright_rows.append(next(spright_iter))
    spright_rows = spright_rows[args.spright_skip:]
    vsr = VsrRelationDataset(
        args.data_root / "vsr" / "annotations" / "zeroshot" / "train.jsonl",
        args.data_root / "vsr" / "vsr_images.zip", image_transform=transform,
    )
    if args.vsr_start + args.vsr_samples > len(vsr):
        raise ValueError(f"VSR range exceeds dataset: start={args.vsr_start} samples={args.vsr_samples} len={len(vsr)}")
    vsr_rows = [vsr[args.vsr_start + index] for index in range(args.vsr_samples)]
    with torch.no_grad():
        spright_images = torch.stack([row["image"] for row in spright_rows])
        vsr_images = torch.stack([row["image"] for row in vsr_rows])
        spright_latents = encode_images(vae, spright_images, device=device, dtype=dtype)
        vsr_latents = encode_images(vae, vsr_images, device=device, dtype=dtype)
        spright_batches = []
        for index, row in enumerate(spright_rows):
            # Reset the transport RNG for every sample so all model states see
            # identical t, x_t, and velocity target tensors.
            set_seed(args.seed + 10000 + index)
            clean = spright_latents[index:index + 1]
            t, x0, x1 = transport.sample(clean)
            t, noisy, target = transport.path_sampler.plan(t, x0, x1)
            spright_batches.append({
                "sample_id": row["sample_id"],
                "spatial_caption": row["spatial_caption"],
                "clean_latent": clean.detach(),
                "t": t.detach(), "noisy_latent": noisy.detach(), "velocity_target": target.detach(),
            })
        vsr_batches = []
        for index, row in enumerate(vsr_rows):
            vsr_batches.append({
                "index": args.vsr_start + index,
                "caption": row["caption"], "label": int(row["label"]),
                "relation": row["relation"], "subj": row["subj"], "obj": row["obj"],
                "clean_latent": vsr_latents[index:index + 1].detach(),
            })
    return spright_batches, vsr_batches


def main() -> None:
    args = parse_args()
    if args.spright_samples < 1 or args.vsr_samples < 1:
        raise ValueError("sample counts must be positive")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("The DRSA gradient probe requires CUDA")
    args.showo_root = args.showo_root.resolve()
    args.config = args.config.resolve()
    args.vae_path = args.vae_path.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.weight_type)
    config = OmegaConf.load(args.config)
    tokenizer, token_ids = __import__("models.misc", fromlist=["get_text_tokenizer"]).get_text_tokenizer(
        str(config.model.showo.llm_model_path), add_showo_tokens=True,
        return_showo_token_ids=True, llm_name="qwen2_5",
    )
    sequence_length = args.max_text_tokens + 734

    from models import Showo2Qwen2_5, WanVAE
    from transport import create_transport

    model = Showo2Qwen2_5.from_pretrained(
        args.model, use_safetensors=False, local_files_only=True,
        low_cpu_mem_usage=False, torch_dtype=dtype, drsa_enabled=True,
    ).to(device=device, dtype=dtype).eval()
    model.requires_grad_(False)
    model.drsa_adapter.requires_grad_(True)
    model.drsa_adapter.eval()
    adapter_initial = copy.deepcopy(model.drsa_adapter.state_dict())
    parameters = list(model.drsa_adapter.parameters())
    groups = module_groups(model.drsa_adapter)
    transport = create_transport(
        path_type=config.transport.path_type, prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight, train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps, snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift, seq_len=729,
    )
    vae = WanVAE(vae_pth=str(args.vae_path), dtype=dtype, device=device)
    spright_batches, vsr_batches = prepare_fixed_batches(
        args, tokenizer, token_ids, config, vae, transport, device, dtype,
    )
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    checkpoints: list[tuple[str, Path | None]] = [("random_init", None), ("step_3000", args.checkpoint_3k), ("step_20000", args.checkpoint_20k)]
    payloads = {label: load_checkpoint(path) for label, path in checkpoints}
    metadata = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed, "spright_samples": len(spright_batches), "vsr_samples": len(vsr_batches),
        "spright": [{"sample_id": row["sample_id"], "caption": row["spatial_caption"]} for row in spright_batches],
        "vsr": [{"index": row["index"], "relation": row["relation"], "label": row["label"]} for row in vsr_batches],
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "groups": {name: sum(parameter.numel() for parameter in values) for name, values in groups.items()},
        "losses": {"spright": "official velocity_prediction L_flow only", "vsr": "official next_token_prediction L_ntp only"},
        "comparison_rule": "U/G are disjoint parameter spaces; no direct U-G cosine is reported. Cosines are only within a shared parameter set.",
    }
    write_json(args.output_dir / "metadata.json", metadata)
    write_json(args.output_dir / "status.json", {"status": "running", "completed": 0, "total": len(checkpoints) * (len(spright_batches) + len(vsr_batches))})

    all_results: list[dict[str, Any]] = []
    total = len(checkpoints) * (len(spright_batches) + len(vsr_batches))
    completed = 0
    for state_name, _ in checkpoints:
        payload = payloads[state_name]
        model.drsa_adapter.load_state_dict(adapter_initial, strict=True)
        if payload is not None:
            model.drsa_adapter.load_state_dict(payload["drsa_adapter"], strict=True)
        optimizer = make_optimizer(model, payload, args)
        state_step = int(payload.get("step", 0)) if payload is not None else 0
        for task_name, batches in (("spright_flow", spright_batches), ("vsr_ntp", vsr_batches)):
            task_gradients: list[dict[int, torch.Tensor]] = []
            task_losses: list[float] = []
            task_rows: list[dict[str, Any]] = []
            for batch in batches:
                model.zero_grad(set_to_none=True)
                if task_name == "spright_flow":
                    loss = evaluate_flow(model, batch, tokenizer, token_ids, transport, device, dtype, args.max_text_tokens, sequence_length)
                else:
                    loss = evaluate_vsr(model, batch, tokenizer, token_ids, device, dtype, args.max_text_tokens, sequence_length)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite {task_name} loss: {float(loss)}")
                gradients = capture_gradients(model, loss, parameters)
                raw_global = math.sqrt(sum(value.square().sum().item() for value in gradients.values()))
                clip_scale = min(1.0, args.grad_clip / raw_global) if raw_global > 0 else 1.0
                metrics = state_metrics(optimizer, groups, parameters, gradients, clip_scale)
                row_id = batch.get("sample_id", batch.get("index"))
                result = {
                    "state": state_name, "checkpoint_step": state_step, "task": task_name,
                    "sample_id": row_id, "loss": float(loss.detach().float().cpu()), "metrics": metrics,
                }
                all_results.append(result)
                task_gradients.append(gradients)
                task_losses.append(result["loss"])
                task_rows.append(result)
                completed += 1
                write_json(args.output_dir / "status.json", {
                    "status": "running", "completed": completed, "total": total,
                    "percent": 100.0 * completed / total, "state": state_name, "task": task_name,
                    "sample_id": row_id, "updated_at": time.time(),
                })
                print(json.dumps({
                    "completed": completed, "total": total, "state": state_name,
                    "task": task_name, "sample_id": row_id, "loss": result["loss"],
                    "u_grad": metrics["understanding"]["raw"]["norm"],
                    "g_grad": metrics["generation"]["raw"]["norm"],
                    "relation_grad": metrics["relation_encoder"]["raw"]["norm"],
                    "clip_scale": metrics["global"]["clip_scale"],
                }, sort_keys=True), flush=True)
                del gradients, loss
                torch.cuda.empty_cache()
            if task_gradients:
                # The task-level mean is computed after averaging squared norms,
                # preserving the scale interpretation across samples.
                summary = {
                    "state": state_name, "checkpoint_step": state_step, "task": task_name,
                    "samples": len(task_rows), "mean_loss": float(np.mean(task_losses)),
                    "mean_metrics": {},
                }
                for group_name, module_parameters in groups.items():
                    indices = {id(parameter): index for index, parameter in enumerate(parameters)}
                    group_indices = [indices[id(parameter)] for parameter in module_parameters]
                    norm_sq = [sum((gradients.get(index).float().square().sum().item() if gradients.get(index) is not None else 0.0) for index in group_indices) for gradients in task_gradients]
                    summary["mean_metrics"][group_name] = {
                        "rms_over_samples": float(math.sqrt(float(np.mean(norm_sq)))),
                        "mean_norm": float(np.mean([math.sqrt(value) for value in norm_sq])),
                        "parameter_rms": task_rows[0]["metrics"][group_name]["raw"].get("parameter_rms", 0.0),
                    }
                all_results.append(summary)

    # Shared relation-encoder task cosine is meaningful; U/G cross-route cosine
    # is intentionally omitted because their coordinates are unrelated.
    comparisons = []
    for state_name, _ in checkpoints:
        flow = [row for row in all_results if row.get("state") == state_name and row.get("task") == "spright_flow" and "metrics" in row]
        vsr = [row for row in all_results if row.get("state") == state_name and row.get("task") == "vsr_ntp" and "metrics" in row]
        if flow and vsr:
            flow_rel = []
            vsr_rel = []
            for row in flow:
                # Sample-level summaries use scalar metrics; retrieve only the
                # same-parameter direction from the saved gradient vectors below.
                del row
            comparisons.append({
                "state": state_name,
                "u_over_g_raw_norm_mean": float(np.mean([row["metrics"]["understanding"]["raw"]["norm"] / max(row["metrics"]["generation"]["raw"]["norm"], 1e-30) for row in flow])),
                "u_over_g_update_norm_mean": float(np.mean([row["metrics"]["understanding"]["adamw_update"]["norm"] / max(row["metrics"]["generation"]["adamw_update"]["norm"], 1e-30) for row in flow])),
                "vsr_g_raw_norm_max": float(max(row["metrics"]["generation"]["raw"]["norm"] for row in vsr)),
                "vsr_u_raw_norm_mean": float(np.mean([row["metrics"]["understanding"]["raw"]["norm"] for row in vsr])),
                "flow_u_raw_norm_mean": float(np.mean([row["metrics"]["understanding"]["raw"]["norm"] for row in flow])),
                "flow_g_raw_norm_mean": float(np.mean([row["metrics"]["generation"]["raw"]["norm"] for row in flow])),
                "relation_cosine": None,
            })
    write_json(args.output_dir / "results.json", {"metadata": metadata, "samples": all_results, "comparisons": comparisons})
    write_json(args.output_dir / "status.json", {
        "status": "completed", "completed": completed, "total": total, "percent": 100.0,
        "updated_at": time.time(), "results": str(args.output_dir / "results.json"),
    })
    print(json.dumps({"status": "completed", "output": str(args.output_dir), "comparisons": comparisons}, indent=2), flush=True)


if __name__ == "__main__":
    main()
