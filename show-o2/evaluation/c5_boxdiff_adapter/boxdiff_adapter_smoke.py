#!/usr/bin/env python3
"""Minimal BoxDiff-style adapter training smoke for Show-o2.

The base model stays frozen. U is supervised through Qwen image-query to
text-key attention; G is supervised through a differentiable phrase-to-image
hidden-state map captured at the generation adapter output. The latter is a
proxy for cross-attention and is deliberately reported as such.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.utils import format_sequence_gen_qwen2_5
from guidance.energy import EnergyConfig, LayoutEnergy
from guidance.layout import LayoutSpec, map_layout_tokens, parse_position_layout
from models import Showo2Qwen2_5, omni_attn_mask_naive
from models.misc import get_text_tokenizer
from utils import path_to_llm_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=WORKSPACE_ROOT / "data/showo2_layout/position_unseen72_v1.jsonl",
    )
    parser.add_argument("--model", default="showlab/show-o2-1.5B")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Optional DRSA-only checkpoint used as the common ablation initialization.",
    )
    parser.add_argument(
        "--route-mode",
        choices=("u_only", "g_only", "u_plus_g"),
        default="u_plus_g",
        help="Train the understanding route, generation route, or both.",
    )
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save every N steps; zero disables step checkpoints.",
    )
    parser.add_argument("--limit-prompts", type=int, default=4)
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--lr-u", type=float, default=1e-5)
    parser.add_argument("--lr-g", type=float, default=5e-6)
    parser.add_argument("--lr-relation", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--u-loss-weight", type=float, default=1.0)
    parser.add_argument("--g-loss-weight", type=float, default=1.0)
    parser.add_argument("--map-temperature", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_manifest(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows[:limit] if limit > 0 else rows


def build_sequence(
    prompt: str,
    tokenizer: Any,
    token_ids: dict[str, int],
    max_text_tokens: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    caption_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_text_tokens,
    )["input_ids"]
    values = format_sequence_gen_qwen2_5(
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
    return tuple(value.unsqueeze(0).to(device) for value in values)


def phrase_embeddings(
    model: Showo2Qwen2_5,
    tokenizer: Any,
    layout: LayoutSpec,
    device: torch.device,
) -> torch.Tensor:
    vectors = []
    with torch.no_grad():
        for item in layout.objects:
            ids = tokenizer(item.phrase, add_special_tokens=False)["input_ids"]
            if not ids:
                raise ValueError(f"empty phrase tokenization: {item.phrase!r}")
            ids_tensor = torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)
            vectors.append(model.showo.model.embed_tokens(ids_tensor)[0].mean(dim=0))
    return torch.stack(vectors).detach()


def mask_for_box(
    box: tuple[float, float, float, float],
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    x0, y0, x1, y1 = box
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return ((xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)).float()


def g_box_energy(
    hidden: torch.Tensor,
    layout: LayoutSpec,
    phrase_vectors: torch.Tensor,
    *,
    height: int,
    width: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return G box energy, relation energy, and JSON-safe diagnostics."""
    if hidden.ndim != 3 or hidden.shape[1] != height * width:
        raise ValueError(f"generation hook has unexpected shape {tuple(hidden.shape)}")
    image = F.normalize(hidden.float(), dim=-1)
    phrases = F.normalize(phrase_vectors.float(), dim=-1)
    logits = torch.einsum("bld,od->blo", image, phrases) * temperature
    maps = logits.softmax(dim=1)[0].transpose(0, 1).reshape(
        len(layout.objects), height, width
    )
    ys = (torch.arange(height, device=hidden.device, dtype=torch.float32) + 0.5) / height
    xs = (torch.arange(width, device=hidden.device, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    box_terms = []
    centroids = []
    details: list[dict[str, Any]] = []
    for index, item in enumerate(layout.objects):
        prob = maps[index]
        mask = mask_for_box(item.box, height, width, hidden.device)
        inside_mass = (prob * mask).sum()
        outside_mass = (prob * (1.0 - mask)).sum()
        normalized = prob / prob.sum().clamp_min(1e-8)
        centroid_x = (normalized * xx).sum()
        centroid_y = (normalized * yy).sum()
        centroids.append((centroid_x, centroid_y))
        projected_x = prob.amax(dim=0)
        projected_y = prob.amax(dim=1)
        target_x = mask.amax(dim=0)
        target_y = mask.amax(dim=1)
        corner = (
            (projected_x - target_x).abs().mean()
            + (projected_y - target_y).abs().mean()
        )
        inside_loss = (1.0 - inside_mass).square()
        outside_loss = outside_mass
        box_terms.append(inside_loss + 0.5 * outside_loss + 0.25 * corner)
        details.append({
            "phrase": item.phrase,
            "inside_mass": float(inside_mass.detach().cpu()),
            "outside_mass": float(outside_mass.detach().cpu()),
            "centroid_x": float(centroid_x.detach().cpu()),
            "centroid_y": float(centroid_y.detach().cpu()),
            "corner_loss": float(corner.detach().cpu()),
        })
    box_loss = torch.stack(box_terms).mean()
    relation_loss = hidden.new_zeros((), dtype=torch.float32)
    if len(centroids) == 2 and layout.relation in {"left", "right", "above", "below"}:
        (x0, y0), (x1, y1) = centroids
        signed = {
            "left": x1 - x0,
            "right": x0 - x1,
            "above": y1 - y0,
            "below": y0 - y1,
        }[layout.relation]
        violation = F.relu(signed.new_tensor(0.20) - signed)
        relation_loss = violation.square()
        for detail in details:
            detail["relation"] = layout.relation
            detail["relation_violation"] = float(violation.detach().cpu())
    return box_loss, relation_loss, {"objects": details}


def adapter_grad_norm(module: torch.nn.Module) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).sum().sqrt().detach().cpu())


def parameter_snapshot(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def parameter_update_norm(
    module: torch.nn.Module,
    before: list[torch.Tensor],
) -> float:
    parameters = list(module.parameters())
    if len(parameters) != len(before):
        raise RuntimeError("parameter snapshot does not match module")
    values = [
        (parameter.detach().float() - old.float()).square().sum()
        for parameter, old in zip(parameters, before)
    ]
    if not values:
        return 0.0
    return float(torch.stack(values).sum().sqrt().cpu())


def save_checkpoint(
    model: Showo2Qwen2_5,
    optimizer: AdamW,
    step: int,
    run_dir: Path,
    metrics: dict[str, Any],
) -> str:
    path = run_dir / "checkpoints" / f"step-{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "drsa_adapter": {
            key: value.detach().cpu()
            for key, value in model.drsa_adapter.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
    }, path)
    return str(path)


def scalarize_details(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for detail in details:
        row = {}
        for key, value in detail.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                row[key] = float(value.detach().cpu())
            else:
                row[key] = value
        converted.append(row)
    return converted


def main() -> int:
    args = parse_args()
    if args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint-every must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("BoxDiff adapter smoke requires CUDA")
    device = torch.device(args.device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.run_dir / "status.json"
    started = time.time()
    atomic_json(status_path, {
        "status": "initializing",
        "phase": "loading_model",
        "step": 0,
        "total_steps": args.max_steps,
        "percent": 0.0,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "run_dir": str(args.run_dir),
    })
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    llm_path = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer, token_ids = get_text_tokenizer(
        llm_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[llm_path],
    )
    sequence_length = args.max_text_tokens + 734
    model = Showo2Qwen2_5.from_pretrained(
        args.model,
        use_safetensors=False,
        local_files_only=True,
        low_cpu_mem_usage=False,
        torch_dtype=torch.bfloat16,
        drsa_enabled=True,
    ).to(device=device, dtype=torch.bfloat16).eval()
    model.requires_grad_(False)
    if model.drsa_adapter is None:
        raise RuntimeError("model was not constructed with drsa_enabled=True")
    model.drsa_adapter.requires_grad_(False)
    if args.route_mode in {"u_only", "u_plus_g"}:
        model.drsa_adapter.understanding.requires_grad_(True)
    if args.route_mode in {"g_only", "u_plus_g"}:
        model.drsa_adapter.generation.requires_grad_(True)
    # Relation conditioning is shared and remains trainable in every route.
    model.drsa_adapter.relation_encoder.requires_grad_(True)
    model.drsa_adapter.train()
    if args.init_checkpoint is not None:
        checkpoint = torch.load(
            args.init_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        state = checkpoint.get("drsa_adapter")
        if not isinstance(state, dict):
            raise ValueError(
                f"checkpoint has no DRSA adapter state: {args.init_checkpoint}"
            )
        if any(key.startswith("generation_spatial.") for key in state):
            model.drsa_adapter.enable_generation_spatial()
        model.drsa_adapter.load_state_dict(state, strict=True)

    parameter_groups = []
    if args.route_mode in {"u_only", "u_plus_g"}:
        parameter_groups.append({
            "params": model.drsa_adapter.understanding.parameters(),
            "lr": args.lr_u,
            "name": "u",
        })
    if args.route_mode in {"g_only", "u_plus_g"}:
        parameter_groups.append({
            "params": model.drsa_adapter.generation.parameters(),
            "lr": args.lr_g,
            "name": "g",
        })
    parameter_groups.append({
        "params": model.drsa_adapter.relation_encoder.parameters(),
        "lr": args.lr_relation,
        "name": "relation",
    })
    optimizer = AdamW(parameter_groups, weight_decay=args.weight_decay)

    prepared = []
    for row in read_manifest(args.manifest, args.limit_prompts):
        prompt = str(row.get("prompt", "")).strip()
        layout = map_layout_tokens(parse_position_layout(prompt), tokenizer)
        prepared.append(
            (prompt, layout, phrase_embeddings(model, tokenizer, layout, device))
        )
    if not prepared:
        raise ValueError("no prompts prepared")

    energy = LayoutEnergy(EnergyConfig(
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
    ))
    hook_state: dict[str, torch.Tensor] = {}

    def generation_hook(_module, _inputs, output):
        hook_state["hidden"] = output[0] if isinstance(output, tuple) else output

    hook_handle = model.drsa_adapter.generation.register_forward_hook(generation_hook)
    step_records: list[dict[str, Any]] = []
    try:
        for step in range(1, args.max_steps + 1):
            prompt, layout, phrase_vectors = prepared[(step - 1) % len(prepared)]
            optimizer.zero_grad(set_to_none=True)
            hook_state.clear()
            (
                text_tokens,
                _labels,
                modality_positions,
                text_mask,
                image_mask,
            ) = build_sequence(
                prompt,
                tokenizer,
                token_ids,
                args.max_text_tokens,
                sequence_length,
                device,
            )
            attention_mask = omni_attn_mask_naive(
                1, sequence_length, modality_positions, device
            ).to(torch.bfloat16)
            relation_text = (
                f"{layout.objects[0].phrase} {layout.relation} "
                f"{layout.objects[1].phrase}"
            )
            encoded = tokenizer(
                [relation_text],
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=32,
                return_tensors="pt",
            )
            with torch.no_grad():
                text_features = model.showo.model.embed_tokens(
                    encoded["input_ids"].to(device)
                )
            relation_code = model.drsa_adapter.encode_relation(
                text_features.detach(), encoded["attention_mask"].to(device)
            )
            latent = torch.randn(
                (1, 16, 54, 54), device=device, dtype=torch.bfloat16
            )
            timestep = torch.full(
                (1,), 0.5, device=device, dtype=torch.bfloat16
            )
            _logits, _velocity, attentions = model(
                text_tokens=text_tokens,
                image_latents=latent,
                t=timestep,
                attention_mask=attention_mask,
                text_masks=text_mask,
                image_masks=image_mask,
                modality_positions=modality_positions,
                output_attentions=True,
                output_hidden_states=True,
                max_seq_len=sequence_length,
                device=device,
                drsa_relation_codes=relation_code,
            )
            if not attentions or all(item is None for item in attentions):
                raise RuntimeError(
                    "Qwen attentions were not returned; BoxDiff U loss cannot run"
                )
            u_total, _per_sample, u_details = energy.evaluate(
                attentions,
                [layout],
                modality_positions,
                (27, 27),
                True,
            )
            u_box = torch.stack(
                [item["boxdiff_energy"] for item in u_details]
            ).float().mean()
            u_relation = torch.stack([
                item.get("relation_energy", u_total.new_zeros(()))
                for item in u_details
            ]).float().mean()
            if "hidden" not in hook_state:
                raise RuntimeError(
                    "generation adapter hook did not capture hidden states"
                )
            g_box, g_relation, g_details = g_box_energy(
                hook_state["hidden"],
                layout,
                phrase_vectors,
                height=27,
                width=27,
                temperature=args.map_temperature,
            )
            u_loss = u_box + u_relation
            g_loss = g_box + g_relation
            loss = (
                args.u_loss_weight * u_loss
                if args.route_mode in {"u_only", "u_plus_g"}
                else u_loss.new_zeros(())
            ) + (
                args.g_loss_weight * g_loss
                if args.route_mode in {"g_only", "u_plus_g"}
                else g_loss.new_zeros(())
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite BoxDiff loss at step {step}: {loss}"
                )
            loss.backward()
            grad_u = adapter_grad_norm(model.drsa_adapter.understanding)
            grad_g = adapter_grad_norm(model.drsa_adapter.generation)
            grad_relation = adapter_grad_norm(model.drsa_adapter.relation_encoder)
            if args.route_mode == "u_only" and grad_g != 0.0:
                raise AssertionError(f"u_only produced G gradient: {grad_g}")
            if args.route_mode == "g_only" and grad_u != 0.0:
                raise AssertionError(f"g_only produced U gradient: {grad_u}")
            before_u = parameter_snapshot(model.drsa_adapter.understanding)
            before_g = parameter_snapshot(model.drsa_adapter.generation)
            before_relation = parameter_snapshot(model.drsa_adapter.relation_encoder)
            if args.route_mode == "u_plus_g" and (grad_u == 0.0 or grad_g == 0.0):
                raise AssertionError(
                    f"u_plus_g did not update both routes: U={grad_u}, G={grad_g}"
                )
            optimizer.step()
            update_u = parameter_update_norm(
                model.drsa_adapter.understanding, before_u
            )
            update_g = parameter_update_norm(
                model.drsa_adapter.generation, before_g
            )
            update_relation = parameter_update_norm(
                model.drsa_adapter.relation_encoder, before_relation
            )
            if args.route_mode == "u_only" and update_g != 0.0:
                raise AssertionError(f"u_only updated G parameters: {update_g}")
            if args.route_mode == "g_only" and update_u != 0.0:
                raise AssertionError(f"g_only updated U parameters: {update_u}")
            if args.route_mode == "u_plus_g" and (update_u == 0.0 or update_g == 0.0):
                raise AssertionError(
                    f"u_plus_g did not update both routes: U={update_u}, G={update_g}"
                )
            elapsed = time.time() - started
            record = {
                "step": step,
                "prompt": prompt,
                "loss": float(loss.detach().cpu()),
                "route_mode": args.route_mode,
                "active_u": args.route_mode in {"u_only", "u_plus_g"},
                "active_g": args.route_mode in {"g_only", "u_plus_g"},
                "loss_box_u": float(u_box.detach().cpu()),
                "loss_relation_u": float(u_relation.detach().cpu()),
                "loss_box_g": float(g_box.detach().cpu()),
                "loss_relation_g": float(g_relation.detach().cpu()),
                "grad_u": grad_u,
                "grad_g": grad_g,
                "grad_relation": grad_relation,
                "update_u": update_u,
                "update_g": update_g,
                "update_relation": update_relation,
                "lr_u": args.lr_u,
                "lr_g": args.lr_g,
                "lr_relation": args.lr_relation,
                "u_details": scalarize_details(u_details),
                "g_details": g_details,
                "elapsed_seconds": elapsed,
            }
            step_records.append(record)
            atomic_json(
                args.run_dir / "gradient_probe.json", {"records": step_records}
            )
            atomic_json(status_path, {
                "status": "running",
                "phase": "boxdiff_adapter_update",
                "step": step,
                "total_steps": args.max_steps,
                "percent": 100.0 * step / args.max_steps,
                "updated_at": now_iso(),
                "elapsed_seconds": elapsed,
                "metrics": record,
            })
            if args.checkpoint_every > 0 and (
                step % args.checkpoint_every == 0 or step == args.max_steps
            ):
                save_checkpoint(model, optimizer, step, args.run_dir, record)
            print(
                f"step={step}/{args.max_steps} loss={record['loss']:.6f} "
                f"u_grad={grad_u:.4e} g_grad={grad_g:.4e} "
                f"u_update={update_u:.4e} g_update={update_g:.4e} "
                f"relation_grad={grad_relation:.4e}",
                flush=True,
            )
        atomic_json(args.run_dir / "metadata.json", {
            "status": "completed",
            "protocol": "boxdiff-adapter-pseudo-box-smoke-v1",
            "route_mode": args.route_mode,
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            "manifest": str(args.manifest),
            "prompt_count": len(prepared),
            "base_model_frozen": True,
            "pseudo_box": True,
            "learning_rates": {
                "u": args.lr_u,
                "g": args.lr_g,
                "relation": args.lr_relation,
            },
            "loss": (
                "U Qwen pseudo-cross-attention + G phrase-to-image hidden map; "
                "inside/outside/corner/relation"
            ),
            "created_at": now_iso(),
        })
        atomic_json(status_path, {
            "status": "completed",
            "phase": "completed",
            "step": args.max_steps,
            "total_steps": args.max_steps,
            "percent": 100.0,
            "updated_at": now_iso(),
            "elapsed_seconds": time.time() - started,
        })
        return 0
    finally:
        hook_handle.remove()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        traceback.print_exc()
        parsed = parse_args()
        parsed.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(parsed.run_dir / "status.json", {
            "status": "failed",
            "updated_at": now_iso(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        })
        raise
