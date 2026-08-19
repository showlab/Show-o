#!/usr/bin/env python3
"""Localize text-conditioning loss across Show-o2's T2I forward path."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


VARIANTS = ("matched", "same", "counterfactual", "null")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--showo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", default=os.environ.get("SHOWO2_MODEL", "showlab/show-o2-1.5B"))
    parser.add_argument("--vae-path", type=Path, default=os.environ.get("SHOWO2_VAE_PATH"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight-type", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--vae-weight-type", choices=("bfloat16", "float32"), default="float32")
    parser.add_argument("--timesteps", type=float, nargs="+", default=(0.2, 0.5, 0.8))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=0)
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


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def clean_float(value: torch.Tensor | float) -> float | None:
    number = float(value.item()) if isinstance(value, torch.Tensor) else float(value)
    return number if math.isfinite(number) else None


def compare_tensor(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | None]:
    a = reference.detach().float().reshape(-1)
    b = candidate.detach().float().reshape(-1)
    delta = a - b
    norm_a = torch.linalg.vector_norm(a)
    norm_b = torch.linalg.vector_norm(b)
    delta_norm = torch.linalg.vector_norm(delta)
    scale = (norm_a + norm_b) * 0.5
    denominator = norm_a * norm_b
    cosine = torch.dot(a, b) / denominator if denominator > 0 else torch.tensor(float("nan"), device=a.device)
    return {
        "relative_l2": clean_float(delta_norm / scale) if scale > 0 else None,
        "cosine": clean_float(cosine),
        "rms_delta": clean_float(delta.square().mean().sqrt()),
        "reference_rms": clean_float(a.square().mean().sqrt()),
    }


def stage_metrics(tensor: torch.Tensor) -> dict:
    if tensor.shape[0] != len(VARIANTS):
        raise RuntimeError(f"Expected {len(VARIANTS)} variants, got shape {tuple(tensor.shape)}")
    matched = tensor[0]
    comparisons = {
        "same": compare_tensor(matched, tensor[1]),
        "counterfactual": compare_tensor(matched, tensor[2]),
        "null": compare_tensor(matched, tensor[3]),
    }
    baseline = comparisons["same"]["relative_l2"]
    for name in ("counterfactual", "null"):
        sensitivity = comparisons[name]["relative_l2"]
        comparisons[name]["condition_gain"] = (
            sensitivity - baseline if sensitivity is not None and baseline is not None else None
        )
    comparisons["same"]["condition_gain"] = 0.0 if baseline is not None else None
    return comparisons


def self_test() -> None:
    tensor = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    metrics = stage_metrics(tensor)
    assert metrics["same"]["relative_l2"] == 0.0
    assert abs(metrics["counterfactual"]["cosine"]) < 1e-8
    assert metrics["counterfactual"]["condition_gain"] > 1.0
    print(json.dumps({"status": "ok", "metrics": metrics}, sort_keys=True))


class ActivationCapture:
    def __init__(self, add_time_embeds: bool):
        self.add_time_embeds = add_time_embeds
        self.modality_positions: torch.Tensor | None = None
        self.values: dict[str, torch.Tensor] = {}
        self.handles = []

    def image_slices(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.modality_positions is None:
            raise RuntimeError("modality_positions were not set before model execution")
        output = []
        for batch_index, modality_batch in enumerate(self.modality_positions):
            offset, length = (int(value.item()) for value in modality_batch[0])
            start = offset + (1 if self.add_time_embeds else 0)
            count = length - (1 if self.add_time_embeds else 0)
            output.append(tensor[batch_index, start:start + count])
        lengths = {item.shape[0] for item in output}
        if len(lengths) != 1:
            raise RuntimeError(f"Inconsistent image-token lengths: {sorted(lengths)}")
        return torch.stack(output).detach()

    def add(self, name: str, module: torch.nn.Module, fusion: bool = False) -> None:
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Hook {name} returned {type(value)!r}")
            self.values[name] = value.detach() if fusion else self.image_slices(value)
        self.handles.append(module.register_forward_hook(hook))

    def clear(self, modality_positions: torch.Tensor) -> None:
        self.values.clear()
        self.modality_positions = modality_positions

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def load_manifest(path: Path, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit > 0 else rows


def record_id(sample_id: str, timestep: float) -> str:
    return f"{sample_id}@t={timestep:.6f}"


def variant_errors(
    velocity: torch.Tensor,
    euler_prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    clean_latent: torch.Tensor,
) -> dict:
    output = {}
    for index, name in enumerate(VARIANTS):
        output[name] = {
            "velocity_target_mse": clean_float(
                (velocity[index].float() - target_velocity[0].float()).square().mean()
            ),
            "euler_clean_mse": clean_float(
                (euler_prediction[index].float() - clean_latent[0].float()).square().mean()
            ),
        }
    for metric in ("velocity_target_mse", "euler_clean_mse"):
        matched = output["matched"][metric]
        for name in ("same", "counterfactual", "null"):
            output[name][f"{metric}_margin_vs_matched"] = output[name][metric] - matched
    return output


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.manifest is None or args.output_dir is None:
        raise ValueError("--manifest and --output-dir are required")
    if not args.timesteps or any(not 0.0 <= value <= 1.0 for value in args.timesteps):
        raise ValueError("--timesteps must contain values in [0, 1]")

    args.showo_root = args.showo_root.resolve()
    args.config = (args.config or args.showo_root / "configs/showo2_1.5b_demo_432x432.yaml").resolve()
    args.vae_path = (args.vae_path or args.showo_root / "Wan2.1_VAE.pth").resolve()
    if not args.manifest.is_file() or not args.config.is_file() or not args.vae_path.is_file():
        raise FileNotFoundError("manifest, config, or VAE checkpoint is missing")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("The real condition probe requires CUDA")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.output_dir / "samples.jsonl"
    status_path = args.output_dir / "status.json"
    rows = load_manifest(args.manifest, args.limit)
    expected = [record_id(row["sample_id"], timestep) for row in rows for timestep in args.timesteps]
    completed = set()
    if args.resume and samples_path.is_file():
        for line in samples_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(json.loads(line)["record_id"])

    started_at = int(time.time())
    status = {
        "status": "loading",
        "pid": os.getpid(),
        "started_at": started_at,
        "updated_at": started_at,
        "total": len(expected),
        "completed": len(completed.intersection(expected)),
        "current_record": None,
        "output": str(samples_path),
        "error": "",
    }
    write_json_atomic(status_path, status)

    root_string = str(args.showo_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    from omegaconf import OmegaConf
    from datasets.utils import format_sequence_gen_qwen2_5, image_transform, remove_prefix
    from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
    from models.misc import get_text_tokenizer

    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)
    config.model.showo.pretrained_model_path = args.model
    weight_type = dtype_from_name(args.weight_type)
    vae_weight_type = dtype_from_name(args.vae_weight_type)
    device = torch.device(args.device)

    seed_everything(args.seed)
    tokenizer, token_ids = get_text_tokenizer(
        str(config.model.showo.llm_model_path),
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name="qwen2_5",
    )
    config.model.showo.llm_vocab_size = len(tokenizer)
    model = Showo2Qwen2_5.from_pretrained(
        args.model,
        use_safetensors=False,
        local_files_only=True,
    ).to(device=device, dtype=weight_type).eval().requires_grad_(False)
    vae = WanVAE(vae_pth=str(args.vae_path), dtype=vae_weight_type, device=device)

    num_image_tokens = int(config.dataset.preprocessing.num_t2i_image_tokens)
    add_time_embeds = bool(config.model.showo.add_time_embeds)
    sequence_image_tokens = num_image_tokens + (1 if add_time_embeds else 0)
    max_seq_len = int(config.dataset.preprocessing.max_seq_length)
    max_text_len = max_seq_len - sequence_image_tokens - 4
    system_tokens = tokenizer(("", "", ""), add_special_tokens=False).input_ids
    system_token_len = sum(len(tokens) for tokens in system_tokens)

    qwen_layers = model.showo.model.layers
    qwen_indices = sorted({0, len(qwen_layers) // 2, len(qwen_layers) - 1})
    capture = ActivationCapture(add_time_embeds=add_time_embeds)
    capture.add("fusion", model.fusion_proj, fusion=True)
    for index in qwen_indices:
        capture.add(f"qwen_layer_{index + 1:02d}", qwen_layers[index])
    capture.add("qwen_final", model.showo.model.norm)
    if hasattr(model, "diff_proj"):
        capture.add("diff_proj", model.diff_proj)
    head_indices = sorted({0, (len(model.diffusion_head_a) - 1) // 2, len(model.diffusion_head_a) - 1})
    for index in head_indices:
        capture.add(f"diffusion_head_{index + 1:02d}", model.diffusion_head_a[index])
    capture.add("velocity_tokens", model.diffusion_head_b)

    metadata = {
        "benchmark": "Show-o2 C2 condition-localization probe",
        "model": args.model,
        "config": str(args.config),
        "vae": str(args.vae_path),
        "manifest": str(args.manifest.resolve()),
        "seed": args.seed,
        "timesteps": list(args.timesteps),
        "variants": list(VARIANTS),
        "stages": ["fusion", *[f"qwen_layer_{i + 1:02d}" for i in qwen_indices], "qwen_final"]
                  + (["diff_proj"] if hasattr(model, "diff_proj") else [])
                  + [f"diffusion_head_{i + 1:02d}" for i in head_indices]
                  + ["velocity_tokens", "velocity", "euler_prediction"],
        "qwen_layers": len(qwen_layers),
        "diffusion_head_layers": len(model.diffusion_head_a),
        "num_spatial_image_tokens": num_image_tokens,
        "sequence_image_tokens": sequence_image_tokens,
        "add_time_embeds": add_time_embeds,
        "control": "matched and same are duplicate batch entries; fixed x0, x1, and t across all variants",
        "euler_prediction": "x_t + (1-t)*v_pred; a one-step local extrapolation toward the data endpoint, not a full sample",
    }
    write_json_atomic(args.output_dir / "metadata.json", metadata)
    status["status"] = "running"
    write_json_atomic(status_path, status)

    try:
        for row_number, row in enumerate(rows):
            pending_timesteps = [
                timestep for timestep in args.timesteps
                if record_id(row["sample_id"], timestep) not in completed
            ]
            if not pending_timesteps:
                continue

            sample_seed = args.seed + int(row.get("index", row_number))
            seed_everything(sample_seed)
            with Image.open(row["image"]) as source:
                image = source.convert("RGB")
            pixel_values = image_transform(image, resolution=int(config.dataset.preprocessing.resolution))
            pixel_values = pixel_values.unsqueeze(0).unsqueeze(2).to(device=device, dtype=vae_weight_type)
            with torch.inference_mode():
                clean_latent = vae.sample(pixel_values, deterministic=True).squeeze(2).to(device)
            noise = torch.randn_like(clean_latent)
            target_velocity = clean_latent - noise

            prompts = (
                row["matched_prompt"],
                row["matched_prompt"],
                row["counterfactual_prompt"],
                "",
            )
            formatted = []
            token_counts = {}
            for name, prompt in zip(VARIANTS, prompts):
                ids = tokenizer(
                    remove_prefix(prompt.replace("\n", "")),
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_text_len,
                ).input_ids
                token_counts[name] = len(ids)
                formatted.append(format_sequence_gen_qwen2_5(
                    ids, system_tokens, token_ids["bos_id"], token_ids["eos_id"], token_ids["boi_id"],
                    token_ids["eoi_id"], tokenizer.pad_token_id, token_ids["img_pad_id"], sequence_image_tokens,
                    max_seq_len, system_token_len,
                ))
            text_tokens = torch.stack([item[0] for item in formatted]).to(device)
            modality_positions = torch.stack([item[2] for item in formatted]).to(device)
            attention_mask = omni_attn_mask_naive(
                len(VARIANTS), max_seq_len, modality_positions, device
            ).to(weight_type)
            repeated_clean = clean_latent.repeat(len(VARIANTS), 1, 1, 1)
            repeated_noise = noise.repeat(len(VARIANTS), 1, 1, 1)

            for timestep in pending_timesteps:
                current_id = record_id(row["sample_id"], timestep)
                record_started = time.time()
                status.update({"current_record": current_id, "updated_at": int(time.time())})
                write_json_atomic(status_path, status)
                t = torch.full((len(VARIANTS),), float(timestep), device=device, dtype=weight_type)
                latent = float(timestep) * repeated_clean + (1.0 - float(timestep)) * repeated_noise
                capture.clear(modality_positions)
                torch.cuda.reset_peak_memory_stats(device)
                with torch.inference_mode():
                    _, velocity = model(
                        text_tokens=text_tokens,
                        image_latents=latent,
                        t=t,
                        attention_mask=attention_mask,
                        modality_positions=modality_positions,
                        output_hidden_states=True,
                        max_seq_len=max_seq_len,
                        device=device,
                    )
                euler_prediction = latent + (1.0 - float(timestep)) * velocity

                stages = {name: stage_metrics(value) for name, value in capture.values.items()}
                stages["velocity"] = stage_metrics(velocity)
                stages["euler_prediction"] = stage_metrics(euler_prediction)
                qwen_sensitivity = stages["qwen_final"]["counterfactual"]["condition_gain"]
                retention = {}
                for name, values in stages.items():
                    gain = values["counterfactual"]["condition_gain"]
                    retention[name] = gain / qwen_sensitivity if gain is not None and qwen_sensitivity not in (None, 0.0) else None

                record = {
                    "record_id": current_id,
                    "sample_id": row["sample_id"],
                    "prompt_id": row["prompt_id"],
                    "index": row.get("index", row_number),
                    "tag": row["tag"],
                    "correct": bool(row["correct"]),
                    "image": row["image"],
                    "prompts": dict(zip(VARIANTS, prompts)),
                    "counterfactual_edit": row["counterfactual_edit"],
                    "prompt_token_counts": token_counts,
                    "modality_offsets": {
                        name: int(modality_positions[index, 0, 0].item())
                        for index, name in enumerate(VARIANTS)
                    },
                    "sample_seed": sample_seed,
                    "timestep": float(timestep),
                    "stages": stages,
                    "counterfactual_retention_vs_qwen_final": retention,
                    "prediction_errors": variant_errors(
                        velocity, euler_prediction, target_velocity, clean_latent
                    ),
                    "runtime_seconds": time.time() - record_started,
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
                }

                with samples_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                completed.add(current_id)
                status.update({
                    "completed": len(completed.intersection(expected)),
                    "current_record": None,
                    "updated_at": int(time.time()),
                    "last_runtime_seconds": record["runtime_seconds"],
                })
                write_json_atomic(status_path, status)
                print(json.dumps({
                    "completed": status["completed"],
                    "total": status["total"],
                    "record_id": current_id,
                    "tag": row["tag"],
                    "qwen_gain": stages["qwen_final"]["counterfactual"]["condition_gain"],
                    "head_gain": stages[f"diffusion_head_{head_indices[-1] + 1:02d}"]["counterfactual"]["condition_gain"],
                    "velocity_gain": stages["velocity"]["counterfactual"]["condition_gain"],
                    "runtime_seconds": record["runtime_seconds"],
                }, sort_keys=True), flush=True)
                del velocity, euler_prediction, stages, record
                capture.values.clear()
                gc.collect()
                torch.cuda.empty_cache()

            del pixel_values, clean_latent, noise, target_velocity, repeated_clean, repeated_noise
            gc.collect()
            torch.cuda.empty_cache()
    except BaseException as exc:
        status.update({"status": "failed", "error": repr(exc), "updated_at": int(time.time())})
        write_json_atomic(status_path, status)
        raise
    finally:
        capture.close()

    status.update({
        "status": "completed",
        "completed": len(completed.intersection(expected)),
        "current_record": None,
        "updated_at": int(time.time()),
        "finished_at": int(time.time()),
    })
    write_json_atomic(status_path, status)


if __name__ == "__main__":
    main()
