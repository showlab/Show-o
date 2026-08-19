#!/usr/bin/env python3
"""Probe where Show-o2 represents and transmits prompt-conditioned layout."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


SHOWO2_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = SHOWO2_ROOT.parents[1]
RELATION_INVERSE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
RELATION_PHRASE = {
    "left": "left of",
    "right": "right of",
    "above": "above",
    "below": "below",
}


def parse_indices(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layer indices must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one layer index is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=WORKSPACE_ROOT / "data/showo2_layout/position_holdout20_v1.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=SHOWO2_ROOT / "configs/showo2_1.5b_demo_432x432.yaml",
    )
    parser.add_argument("--model", default=os.environ.get("SHOWO2_MODEL", "showlab/show-o2-1.5B"))
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument("--weight-type", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timesteps", type=float, nargs="+", default=(0.1, 0.3, 0.5))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Spread --limit across left/right/above/below instead of taking a prefix.",
    )
    parser.add_argument("--relation-margin", type=float, default=0.20)
    parser.add_argument("--boxdiff-p", type=float, default=0.20)
    parser.add_argument("--boxdiff-corner-radius", type=int, default=1)
    parser.add_argument("--boxdiff-text-scale", type=float, default=100.0)
    parser.add_argument("--gradient-qwen-layers", type=parse_indices, default=(0, 14, -1))
    parser.add_argument("--gradient-diffusion-layers", type=parse_indices, default=(0, 4, -1))
    parser.add_argument(
        "--gradient-objective",
        choices=("relation", "boxdiff", "combined"),
        default="combined",
    )
    parser.add_argument("--no-gradients", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_relation(value: str) -> str:
    normalized = value.strip().lower()
    return {
        "left of": "left",
        "right of": "right",
        "left": "left",
        "right": "right",
        "above": "above",
        "below": "below",
    }[normalized]


def invert_prompt(prompt: str, relation: str) -> str:
    source = RELATION_PHRASE[relation]
    target = RELATION_PHRASE[RELATION_INVERSE[relation]]
    inverted, replacements = re.subn(
        rf"\b{re.escape(source)}\b",
        target,
        prompt,
        count=1,
        flags=re.IGNORECASE,
    )
    if replacements != 1:
        raise ValueError(f"Could not invert {source!r} in prompt {prompt!r}")
    return inverted


def select_rows(rows: list[dict[str, Any]], limit: int, balanced: bool) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    if not balanced:
        return rows[:limit]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[normalize_relation(row["relation"])].append(row)
    selected = []
    while len(selected) < limit:
        made_progress = False
        for relation in ("left", "right", "above", "below"):
            if grouped[relation] and len(selected) < limit:
                selected.append(grouped[relation].pop(0))
                made_progress = True
        if not made_progress:
            break
    return selected


def resolve_indices(raw_indices: Sequence[int], layer_count: int) -> tuple[int, ...]:
    resolved = []
    for raw_index in raw_indices:
        index = raw_index if raw_index >= 0 else layer_count + raw_index
        if not 0 <= index < layer_count:
            raise IndexError(f"Layer {raw_index} is outside a stack with {layer_count} layers")
        if index not in resolved:
            resolved.append(index)
    return tuple(resolved)


def clean_float(value: torch.Tensor | float) -> float | None:
    number = float(value.detach().float().cpu()) if isinstance(value, torch.Tensor) else float(value)
    return number if math.isfinite(number) else None


def tensor_rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().cpu())


def tensor_cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_flat = left.detach().float().reshape(-1)
    right_flat = right.detach().float().reshape(-1)
    denominator = torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat)
    if float(denominator) <= 1e-12:
        return None
    return clean_float(torch.dot(left_flat, right_flat) / denominator)


class ActivationCapture:
    def __init__(self) -> None:
        self.values: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []

    def add(self, name: str, module: torch.nn.Module) -> None:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Activation {name} has unsupported type {type(value)!r}")
            self.values[name] = value

        self.handles.append(module.register_forward_hook(hook))

    def clear(self) -> None:
        self.values.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _spatial_map(
    layer_attention: torch.Tensor,
    batch_index: int,
    image_start: int,
    image_end: int,
    token_indices: Sequence[int],
) -> torch.Tensor:
    indices = torch.tensor(token_indices, device=layer_attention.device, dtype=torch.long)
    block = layer_attention[batch_index, :, image_start:image_end, :].index_select(-1, indices)
    return block.float().mean(dim=(0, 2))


def attention_layer_metrics(
    layer_attention: torch.Tensor,
    layouts: Sequence[Any],
    modality_positions: torch.Tensor,
    spatial_shape: tuple[int, int],
    add_time_embeds: bool,
    boxdiff_config: dict[str, Any],
) -> dict[str, Any]:
    from guidance.energy import EnergyConfig, LayoutEnergy

    height, width = spatial_shape
    xs = ((torch.arange(width, device=layer_attention.device).float() + 0.5) / width).repeat(height)
    ys = ((torch.arange(height, device=layer_attention.device).float() + 0.5) / height).repeat_interleave(width)
    energy_fn = LayoutEnergy(
        EnergyConfig(
            layers=(0,),
            inside_weight=0.0,
            relation_weight=1.0,
            relation_margin=boxdiff_config["relation_margin"],
            boxdiff_inner_weight=1.0,
            boxdiff_outer_weight=1.0,
            boxdiff_corner_weight=1.0,
            boxdiff_p=boxdiff_config["p"],
            boxdiff_corner_radius=boxdiff_config["corner_radius"],
            boxdiff_text_scale=boxdiff_config["text_scale"],
        )
    )
    _, details = energy_fn(
        (layer_attention,),
        layouts,
        modality_positions,
        spatial_shape,
        add_time_embeds,
    )
    by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        by_batch[int(detail["batch_index"])].append(detail)

    variants = []
    raw_axis_deltas = []
    for batch_index, layout in enumerate(layouts):
        offset, length = (int(value.item()) for value in modality_positions[batch_index, 0])
        image_start = offset + (1 if add_time_embeds else 0)
        image_count = length - (1 if add_time_embeds else 0)
        image_end = image_start + image_count
        text_mass = layer_attention[
            batch_index, :, image_start:image_end, 1:offset - 1
        ].float().sum(dim=-1).mean()
        object_metrics = []
        for item in layout.objects:
            spatial = _spatial_map(
                layer_attention,
                batch_index,
                image_start,
                image_end,
                item.token_indices,
            )
            normalized = spatial / spatial.sum().clamp_min(1e-12)
            entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum()
            object_metrics.append(
                {
                    "centroid_x": clean_float((normalized * xs).sum()),
                    "centroid_y": clean_float((normalized * ys).sum()),
                    "spatial_entropy": clean_float(entropy),
                    "entity_attention_mass": clean_float(spatial.mean()),
                }
            )
        first, second = object_metrics
        raw_dx = first["centroid_x"] - second["centroid_x"]
        raw_dy = first["centroid_y"] - second["centroid_y"]
        relation = layout.relation
        raw_axis = raw_dx if relation in {"left", "right"} else raw_dy
        raw_axis_deltas.append(raw_axis)
        detail_rows = sorted(by_batch[batch_index], key=lambda item: int(item["object_index"]))
        signed_delta = clean_float(detail_rows[0]["relation_signed_delta"])
        variants.append(
            {
                "variant": "matched" if batch_index == 0 else "inverse",
                "relation": relation,
                "text_mass": clean_float(text_mass),
                "raw_dx": raw_dx,
                "raw_dy": raw_dy,
                "relation_signed_delta": signed_delta,
                "relation_violation": clean_float(detail_rows[0]["relation_violation"]),
                "direction_correct": bool(signed_delta > 0.0),
                "margin_correct": bool(signed_delta >= boxdiff_config["relation_margin"]),
                "objects": object_metrics,
                "boxdiff_inner_loss": sum(clean_float(item["inner_loss"]) for item in detail_rows),
                "boxdiff_outer_loss": sum(clean_float(item["outer_loss"]) for item in detail_rows),
                "boxdiff_corner_loss": sum(clean_float(item["corner_loss"]) for item in detail_rows),
            }
        )
    return {
        "variants": variants,
        "pair_direction_correct": all(item["direction_correct"] for item in variants),
        "pair_margin_correct": all(item["margin_correct"] for item in variants),
        "relation_flip": bool(raw_axis_deltas[0] * raw_axis_deltas[1] < 0.0),
    }


def objective_weights(kind: str) -> dict[str, float]:
    return {
        "relation": {
            "relation_weight": 1.0,
            "boxdiff_inner_weight": 0.0,
            "boxdiff_outer_weight": 0.0,
            "boxdiff_corner_weight": 0.0,
        },
        "boxdiff": {
            "relation_weight": 0.0,
            "boxdiff_inner_weight": 1.0,
            "boxdiff_outer_weight": 1.0,
            "boxdiff_corner_weight": 1.0,
        },
        "combined": {
            "relation_weight": 1.0,
            "boxdiff_inner_weight": 1.0,
            "boxdiff_outer_weight": 1.0,
            "boxdiff_corner_weight": 1.0,
        },
    }[kind]


def gradient_attribution(
    attentions: Sequence[torch.Tensor],
    layer_index: int,
    layouts: Sequence[Any],
    modality_positions: torch.Tensor,
    spatial_shape: tuple[int, int],
    add_time_embeds: bool,
    latent: torch.Tensor,
    capture: ActivationCapture,
    objective_kind: str,
    boxdiff_config: dict[str, Any],
    und_final_name: str,
) -> dict[str, Any]:
    from guidance.energy import EnergyConfig, LayoutEnergy

    weights = objective_weights(objective_kind)
    energy_fn = LayoutEnergy(
        EnergyConfig(
            layers=(layer_index,),
            inside_weight=0.0,
            relation_margin=boxdiff_config["relation_margin"],
            boxdiff_p=boxdiff_config["p"],
            boxdiff_corner_radius=boxdiff_config["corner_radius"],
            boxdiff_text_scale=boxdiff_config["text_scale"],
            **weights,
        )
    )
    energy, _ = energy_fn(
        attentions,
        layouts,
        modality_positions,
        spatial_shape,
        add_time_embeds,
    )
    objective = energy * len(layouts)
    names = list(capture.values)
    activation_values = [capture.values[name] for name in names]
    gradients = torch.autograd.grad(
        objective,
        [*activation_values, latent],
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    activation_gradients = dict(zip(names, gradients[:-1]))
    total_latent_gradient = gradients[-1]
    activations = {}
    for name, value in capture.values.items():
        gradient = activation_gradients[name]
        activations[name] = {
            "activation_rms": tensor_rms(value),
            "gradient_rms": tensor_rms(gradient) if gradient is not None else 0.0,
            "gradient_times_activation_rms": (
                tensor_rms(gradient.float() * value.float()) if gradient is not None else 0.0
            ),
        }

    def branch_vjp(name: str) -> torch.Tensor:
        gradient = activation_gradients[name]
        if gradient is None:
            return torch.zeros_like(latent)
        return torch.autograd.grad(
            capture.values[name],
            latent,
            grad_outputs=gradient,
            retain_graph=True,
            create_graph=False,
        )[0]

    und_latent_gradient = branch_vjp(und_final_name)
    gen_latent_gradient = branch_vjp("gen_embed")
    reconstructed = und_latent_gradient + gen_latent_gradient
    residual = total_latent_gradient - reconstructed
    return {
        "objective": objective_kind,
        "energy": clean_float(energy),
        "latent_gradient_rms": tensor_rms(total_latent_gradient),
        "und_latent_gradient_rms": tensor_rms(und_latent_gradient),
        "gen_latent_gradient_rms": tensor_rms(gen_latent_gradient),
        "und_gen_cosine": tensor_cosine(und_latent_gradient, gen_latent_gradient),
        "und_total_cosine": tensor_cosine(und_latent_gradient, total_latent_gradient),
        "gen_total_cosine": tensor_cosine(gen_latent_gradient, total_latent_gradient),
        "branch_reconstruction_relative_rms": (
            tensor_rms(residual) / max(tensor_rms(total_latent_gradient), 1e-12)
        ),
        "activations": activations,
    }


def self_test() -> None:
    rows = [
        {"relation": relation, "id": index}
        for relation in ("left of", "left of", "right of", "above", "below")
        for index in (0,)
    ]
    selected = select_rows(rows, 4, True)
    assert [normalize_relation(row["relation"]) for row in selected] == [
        "left",
        "right",
        "above",
        "below",
    ]
    assert invert_prompt("a cube left of a ball", "left") == "a cube right of a ball"
    assert resolve_indices((0, -1, 0), 10) == (0, 9)
    print(json.dumps({"status": "ok"}, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if any(not 0.0 <= timestep <= 1.0 for timestep in args.timesteps):
        raise ValueError("timesteps must be in [0, 1]")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The real layout layer probe requires CUDA")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.output_dir / "samples.jsonl"
    status_path = args.output_dir / "status.json"
    rows = select_rows(read_jsonl(args.manifest), args.limit, args.balanced)
    expected_ids = [
        f"{int(row.get('source_index', index))}@t={timestep:.3f}"
        for index, row in enumerate(rows)
        for timestep in args.timesteps
    ]
    completed = set()
    if args.resume and samples_path.is_file():
        completed = {
            json.loads(line)["record_id"]
            for line in samples_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    status = {
        "status": "loading",
        "pid": os.getpid(),
        "completed": len(completed.intersection(expected_ids)),
        "total": len(expected_ids),
        "current": None,
        "updated_at": int(time.time()),
        "error": "",
    }
    write_json_atomic(status_path, status)

    root_string = str(SHOWO2_ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    from omegaconf import OmegaConf

    from guidance import map_layout_tokens, parse_position_layout
    from models import Showo2Qwen2_5, omni_attn_mask_naive
    from models.misc import get_text_tokenizer, prepare_gen_input
    from utils import get_hyper_params, path_to_llm_name

    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)
    weight_type = dtype_from_name(args.weight_type)
    device = torch.device(args.device)
    seed_everything(args.seed)
    tokenizer, token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
    )
    config.model.showo.llm_vocab_size = len(tokenizer)
    add_time_embeds = bool(config.model.showo.add_time_embeds)
    if add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
    hyper = get_hyper_params(config, tokenizer, token_ids)
    (
        num_t2i_image_tokens,
        _num_mmu_image_tokens,
        _num_video_tokens,
        max_seq_len,
        max_text_len,
        image_latent_dim,
        patch_size,
        latent_width,
        latent_height,
        pad_id,
        bos_id,
        eos_id,
        boi_id,
        eoi_id,
        _bov_id,
        _eov_id,
        img_pad_id,
        _vid_pad_id,
        _cfg_scale,
    ) = hyper
    model = (
        Showo2Qwen2_5.from_pretrained(
            args.model,
            use_safetensors=False,
            local_files_only=True,
        )
        .to(device=device, dtype=weight_type)
        .eval()
        .requires_grad_(False)
    )
    qwen_gradient_indices = resolve_indices(args.gradient_qwen_layers, len(model.showo.model.layers))
    diffusion_gradient_indices = resolve_indices(
        args.gradient_diffusion_layers,
        len(model.diffusion_head_a),
    )
    capture = ActivationCapture()
    capture.add("und_patch", model.image_embedder_und)
    capture.add("gen_embed", model.image_embedder_gen)
    for index, layer in enumerate(model.und_trans.layers):
        capture.add(f"und_layer_{index + 1:02d}", layer)
    und_final_name = f"und_layer_{len(model.und_trans.layers):02d}"
    capture.add("fusion", model.fusion_proj)
    boxdiff_config = {
        "relation_margin": args.relation_margin,
        "p": args.boxdiff_p,
        "corner_radius": args.boxdiff_corner_radius,
        "text_scale": args.boxdiff_text_scale,
    }
    metadata = {
        "benchmark": "Show-o2 prompt-conditioned spatial layer and branch attribution probe",
        "model": args.model,
        "manifest": str(args.manifest.resolve()),
        "rows": len(rows),
        "timesteps": list(args.timesteps),
        "seed": args.seed,
        "qwen_layers": len(model.showo.model.layers),
        "diffusion_layers": len(model.diffusion_head_a),
        "und_layers": len(model.und_trans.layers),
        "gradient_qwen_layers": list(qwen_gradient_indices),
        "gradient_diffusion_layers": list(diffusion_gradient_indices),
        "gradient_objective": args.gradient_objective,
        "gradients_enabled": not args.no_gradients,
        "boxdiff": boxdiff_config,
        "control": "matched and inverse prompts share the same latent; both requested directions must be correct and spatial centroids must flip",
        "interpretation": {
            "und_gen": "UND and GEN cannot read the prompt before Fusion; their VJP values measure downstream layout-energy transport, not independent relation parsing.",
            "qwen_diffusion": "Per-layer correctness measures prompt-conditioned image-query to entity-token attention, not final detector accuracy.",
        },
    }
    write_json_atomic(args.output_dir / "metadata.json", metadata)
    status["status"] = "running"
    write_json_atomic(status_path, status)

    try:
        for row_index, row in enumerate(rows):
            relation = normalize_relation(row["relation"])
            prompts = (row["prompt"], invert_prompt(row["prompt"], relation))
            layouts = tuple(
                map_layout_tokens(parse_position_layout(prompt), tokenizer)
                for prompt in prompts
            )
            text_tokens, _, modality_positions, _ = prepare_gen_input(
                prompts,
                tokenizer,
                num_t2i_image_tokens,
                bos_id,
                eos_id,
                boi_id,
                eoi_id,
                pad_id,
                img_pad_id,
                max_text_len,
                device,
            )
            attention_mask = omni_attn_mask_naive(
                len(prompts),
                max_seq_len,
                modality_positions,
                device,
            ).to(weight_type)
            sample_seed = args.seed + int(row.get("source_index", row_index))
            seed_everything(sample_seed)
            base_latent = torch.randn(
                (1, image_latent_dim, latent_height * patch_size, latent_width * patch_size),
                device=device,
                dtype=weight_type,
            )
            for timestep in args.timesteps:
                record_id = f"{int(row.get('source_index', row_index))}@t={timestep:.3f}"
                if record_id in completed:
                    continue
                status.update({"current": record_id, "updated_at": int(time.time())})
                write_json_atomic(status_path, status)
                latent = base_latent.repeat(len(prompts), 1, 1, 1).detach().requires_grad_(True)
                times = torch.full((len(prompts),), timestep, device=device, dtype=weight_type)
                capture.clear()
                torch.cuda.reset_peak_memory_stats(device)
                started = time.time()
                velocity, qwen_attentions, diffusion_attentions = model.t2i_generate_with_all_attentions(
                    image_latents=latent,
                    t=times,
                    text_tokens=text_tokens,
                    attention_mask=attention_mask,
                    modality_positions=modality_positions,
                    max_seq_len=max_seq_len,
                    guidance_scale=0.0,
                )
                layer_metrics = {
                    "qwen": [
                        attention_layer_metrics(
                            attention,
                            layouts,
                            modality_positions,
                            (latent_height, latent_width),
                            add_time_embeds,
                            boxdiff_config,
                        )
                        for attention in qwen_attentions
                    ],
                    "diffusion": [
                        attention_layer_metrics(
                            attention,
                            layouts,
                            modality_positions,
                            (latent_height, latent_width),
                            add_time_embeds,
                            boxdiff_config,
                        )
                        for attention in diffusion_attentions
                    ],
                }
                gradients: dict[str, dict[str, Any]] = {"qwen": {}, "diffusion": {}}
                if not args.no_gradients:
                    for source, attentions, indices in (
                        ("qwen", qwen_attentions, qwen_gradient_indices),
                        ("diffusion", diffusion_attentions, diffusion_gradient_indices),
                    ):
                        for layer_index in indices:
                            gradients[source][str(layer_index)] = gradient_attribution(
                                attentions,
                                layer_index,
                                layouts,
                                modality_positions,
                                (latent_height, latent_width),
                                add_time_embeds,
                                latent,
                                capture,
                                args.gradient_objective,
                                boxdiff_config,
                                und_final_name,
                            )
                record = {
                    "record_id": record_id,
                    "source_index": int(row.get("source_index", row_index)),
                    "local_index": int(row.get("local_index", row_index)),
                    "sample_seed": sample_seed,
                    "timestep": timestep,
                    "prompts": {"matched": prompts[0], "inverse": prompts[1]},
                    "relations": {"matched": layouts[0].relation, "inverse": layouts[1].relation},
                    "layer_metrics": layer_metrics,
                    "gradient_attribution": gradients,
                    "velocity_rms": tensor_rms(velocity),
                    "runtime_seconds": time.time() - started,
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
                }
                with samples_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                completed.add(record_id)
                status.update(
                    {
                        "completed": len(completed.intersection(expected_ids)),
                        "current": None,
                        "updated_at": int(time.time()),
                    }
                )
                write_json_atomic(status_path, status)
                print(
                    json.dumps(
                        {
                            "record": record_id,
                            "completed": status["completed"],
                            "total": status["total"],
                            "seconds": round(record["runtime_seconds"], 2),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                del latent, velocity, qwen_attentions, diffusion_attentions
                gc.collect()
                torch.cuda.empty_cache()
        status.update({"status": "completed", "current": None, "updated_at": int(time.time())})
        write_json_atomic(status_path, status)
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": int(time.time()),
            }
        )
        write_json_atomic(status_path, status)
        raise
    finally:
        capture.close()


if __name__ == "__main__":
    main()
