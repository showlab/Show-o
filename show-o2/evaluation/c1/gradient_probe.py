#!/usr/bin/env python3
"""Measure pairwise task-gradient interference inside a Show-o2 checkpoint."""

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
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image


PAIR_NAMES = ("mmu_t2i", "lm_mmu", "lm_t2i")


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
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def atomic_group(name: str) -> Optional[str]:
    if name.startswith("fusion_proj."):
        return "fusion"
    if name.startswith("image_embedder_und."):
        return "vision.semantic_embed"
    if name.startswith("position_embedding."):
        return "vision.position"
    match = re.match(r"und_trans\.layers\.(\d+)\.", name)
    if match:
        return f"vision.semantic_layer.{int(match.group(1)):02d}"
    if name.startswith("und_trans."):
        return "vision.semantic_other"
    if name.startswith("image_embedder_gen."):
        return "vision.detail_embed"
    if name.startswith("time_embed_proj."):
        return "time.proj"
    if name.startswith("time_embed."):
        return "time.embed"
    if name.startswith("showo.model.embed_tokens."):
        return "qwen.embed"
    match = re.match(r"showo\.model\.layers\.(\d+)\.(.*)", name)
    if match:
        layer = int(match.group(1))
        suffix = match.group(2)
        if suffix.startswith("self_attn."):
            part = "attn"
        elif suffix.startswith("mlp."):
            part = "mlp"
        else:
            part = "norm"
        return f"qwen.layer.{layer:02d}.{part}"
    if name.startswith("showo.model.norm."):
        return "qwen.final_norm"
    return None


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


def raw_terms(grads_a, grads_b, group_indices: dict[str, list[int]], params: list[torch.nn.Parameter]) -> dict:
    output = {}
    for group, indices in group_indices.items():
        norm_a_sq = torch.zeros((), device=params[0].device, dtype=torch.float64)
        norm_b_sq = torch.zeros_like(norm_a_sq)
        dot = torch.zeros_like(norm_a_sq)
        used_a = used_b = shared = 0
        for index in indices:
            grad_a, grad_b = grads_a[index], grads_b[index]
            if grad_a is not None:
                norm_a_sq += grad_a.detach().double().square().sum()
                used_a += params[index].numel()
            if grad_b is not None:
                norm_b_sq += grad_b.detach().double().square().sum()
                used_b += params[index].numel()
            if grad_a is not None and grad_b is not None:
                dot += (grad_a.detach().double() * grad_b.detach().double()).sum()
                shared += params[index].numel()
        output[group] = {
            "norm_a_sq": float(norm_a_sq.item()),
            "norm_b_sq": float(norm_b_sq.item()),
            "dot": float(dot.item()),
            "parameter_count": sum(params[index].numel() for index in indices),
            "used_a": used_a,
            "used_b": used_b,
            "shared": shared,
        }
    return output


def finalize_terms(terms: Iterable[dict]) -> dict:
    combined = defaultdict(float)
    for term in terms:
        for key in ("norm_a_sq", "norm_b_sq", "dot", "parameter_count", "used_a", "used_b", "shared"):
            combined[key] += term[key]
    norm_a = math.sqrt(max(combined["norm_a_sq"], 0.0))
    norm_b = math.sqrt(max(combined["norm_b_sq"], 0.0))
    denominator = norm_a * norm_b
    cosine = combined["dot"] / denominator if denominator else None
    sum_norm_sq = max(combined["norm_a_sq"] + combined["norm_b_sq"] + 2.0 * combined["dot"], 0.0)
    cancellation = 1.0 - math.sqrt(sum_norm_sq) / (norm_a + norm_b) if norm_a + norm_b else None
    return {
        "cosine": cosine,
        "dot": combined["dot"],
        "conflict": bool(combined["dot"] < 0.0) if denominator else None,
        "grad_norm_a": norm_a,
        "grad_norm_b": norm_b,
        "dominance_a_over_b": norm_a / norm_b if norm_b else None,
        "cancellation_ratio": cancellation,
        "parameter_count": int(combined["parameter_count"]),
        "used_a": int(combined["used_a"]),
        "used_b": int(combined["used_b"]),
        "shared": int(combined["shared"]),
    }


def scope_members(groups: Iterable[str]) -> dict[str, list[str]]:
    groups = sorted(groups)
    qwen_parts = [group for group in groups if group.startswith("qwen.")]
    qwen_layers = sorted({".".join(group.split(".")[:3]) for group in qwen_parts if group.startswith("qwen.layer.")})
    vision = [group for group in groups if group.startswith("vision.")]
    time_groups = [group for group in groups if group.startswith("time.")]
    scopes = {group: [group] for group in groups}
    for layer in qwen_layers:
        scopes[layer] = [group for group in groups if group.startswith(layer + ".")]
    layer_numbers = sorted(int(layer.split(".")[2]) for layer in qwen_layers)
    if layer_numbers:
        cut_a = len(layer_numbers) // 3
        cut_b = 2 * len(layer_numbers) // 3
        thirds = {
            "qwen.early": layer_numbers[:cut_a],
            "qwen.middle": layer_numbers[cut_a:cut_b],
            "qwen.late": layer_numbers[cut_b:],
        }
        for scope, numbers in thirds.items():
            scopes[scope] = [
                group for group in qwen_parts
                if any(group.startswith(f"qwen.layer.{number:02d}.") for number in numbers)
            ]
    scopes["qwen.all"] = qwen_parts
    scopes["vision.all"] = vision
    scopes["time.all"] = time_groups
    scopes["shared.all"] = groups
    return {name: members for name, members in scopes.items() if members}


def pair_metrics(grads_a, grads_b, groups, scopes, params) -> dict:
    raw = raw_terms(grads_a, grads_b, groups, params)
    return {
        scope: finalize_terms(raw[group] for group in members)
        for scope, members in scopes.items()
    }


def self_test() -> None:
    params = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(1))]
    groups = {"fusion": [0], "qwen.embed": [1]}
    scopes = scope_members(groups)
    a = (torch.tensor([1.0, 0.0]), torch.tensor([2.0]))
    b = (torch.tensor([-1.0, 0.0]), torch.tensor([2.0]))
    metrics = pair_metrics(a, b, groups, scopes, params)
    assert abs(metrics["fusion"]["cosine"] + 1.0) < 1e-8
    assert metrics["fusion"]["conflict"] is True
    assert abs(metrics["qwen.all"]["cosine"] - 1.0) < 1e-8
    print(json.dumps({"status": "ok", "metrics": metrics}, sort_keys=True))


def load_manifest(path: Path, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit > 0 else rows


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.manifest is None or args.output_dir is None:
        raise ValueError("--manifest and --output-dir are required")
    args.showo_root = args.showo_root.resolve()
    args.config = (args.config or args.showo_root / "configs/showo2_1.5b_demo_432x432.yaml").resolve()
    if args.vae_path is None:
        args.vae_path = args.showo_root / "Wan2.1_VAE.pth"
    args.vae_path = args.vae_path.resolve()
    if not args.manifest.is_file() or not args.config.is_file() or not args.vae_path.is_file():
        raise FileNotFoundError("manifest, config, or VAE checkpoint is missing")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("The real gradient probe requires CUDA")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "samples.jsonl"
    status_path = args.output_dir / "status.json"
    rows = load_manifest(args.manifest, args.limit)
    completed = set()
    if args.resume and result_path.is_file():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(json.loads(line)["sample_id"])

    started_at = int(time.time())
    status = {
        "status": "loading",
        "pid": os.getpid(),
        "started_at": started_at,
        "updated_at": started_at,
        "total": len(rows),
        "completed": len(completed),
        "current_sample": None,
        "output": str(result_path),
        "error": "",
    }
    write_json_atomic(status_path, status)

    root_string = str(args.showo_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    from omegaconf import OmegaConf
    from datasets.utils import format_sequence_gen_qwen2_5, format_sequence_und, image_transform, remove_prefix
    from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
    from models.misc import get_text_tokenizer, next_token_prediction
    from transport import create_transport

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
    ).to(device=device, dtype=weight_type).eval()
    model.requires_grad_(False)

    names, params, group_indices = [], [], defaultdict(list)
    for name, parameter in model.named_parameters():
        group = atomic_group(name)
        if group is None:
            continue
        parameter.requires_grad_(True)
        group_indices[group].append(len(params))
        names.append(name)
        params.append(parameter)
    if "fusion" not in group_indices or not any(group.startswith("qwen.layer.") for group in group_indices):
        raise RuntimeError("Could not identify Fusion and Qwen parameters in the checkpoint")
    scopes = scope_members(group_indices)

    vae = WanVAE(
        vae_pth=str(args.vae_path),
        dtype=vae_weight_type,
        device=device,
    )
    num_image_tokens = int(config.dataset.preprocessing.num_t2i_image_tokens)
    if bool(config.model.showo.add_time_embeds):
        num_image_tokens += 1
    max_seq_len = int(config.dataset.preprocessing.max_seq_length)
    max_text_len = max_seq_len - num_image_tokens - 4
    system_tokens = tokenizer(("", "", ""), add_special_tokens=False).input_ids
    system_token_len = sum(len(tokens) for tokens in system_tokens)
    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=num_image_tokens,
    )

    metadata = {
        "model": args.model,
        "config": str(args.config),
        "vae": str(args.vae_path),
        "manifest": str(args.manifest.resolve()),
        "seed": args.seed,
        "weight_type": args.weight_type,
        "max_seq_len": max_seq_len,
        "num_image_tokens": num_image_tokens,
        "parameter_count": sum(parameter.numel() for parameter in params),
        "groups": {group: sum(params[index].numel() for index in indices) for group, indices in group_indices.items()},
        "pairs": {
            "mmu_t2i": ["image-conditioned next-token prediction", "text-to-image flow matching"],
            "lm_mmu": ["text-only next-token prediction", "image-conditioned next-token prediction"],
            "lm_t2i": ["text-only next-token prediction", "text-to-image flow matching"],
        },
    }
    write_json_atomic(args.output_dir / "metadata.json", metadata)

    status["status"] = "running"
    write_json_atomic(status_path, status)
    try:
        for row_number, row in enumerate(rows):
            if row["sample_id"] in completed:
                continue
            sample_started = time.time()
            sample_seed = args.seed + int(row.get("index", row_number))
            seed_everything(sample_seed)
            status.update({"current_sample": row["sample_id"], "updated_at": int(time.time())})
            write_json_atomic(status_path, status)

            with Image.open(row["image"]) as source:
                image = source.convert("RGB")
            pixel_values = image_transform(image, resolution=int(config.dataset.preprocessing.resolution))
            pixel_values = pixel_values.unsqueeze(0).unsqueeze(2).to(device=device, dtype=vae_weight_type)
            with torch.no_grad():
                clean_latent = vae.sample(pixel_values).squeeze(2)

            prompt = row.get("base_prompt", row["prompt"]).replace("\n", "")
            mmu_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_text_len).input_ids
            t2i_ids = tokenizer(remove_prefix(prompt), add_special_tokens=False, truncation=True, max_length=max_text_len).input_ids
            t2i = format_sequence_gen_qwen2_5(
                t2i_ids, system_tokens, token_ids["bos_id"], token_ids["eos_id"], token_ids["boi_id"],
                token_ids["eoi_id"], tokenizer.pad_token_id, token_ids["img_pad_id"], num_image_tokens,
                max_seq_len, system_token_len,
            )
            mmu = format_sequence_und(
                mmu_ids, token_ids["bos_id"], token_ids["eos_id"], token_ids["boi_id"], token_ids["eoi_id"],
                tokenizer.pad_token_id, token_ids["img_pad_id"], num_image_tokens, max_seq_len,
            )
            text_tokens = torch.stack([t2i[0], mmu[0]]).to(device)
            text_labels = torch.stack([t2i[1], mmu[1]]).to(device)
            modality_positions = torch.stack([t2i[2], mmu[2]]).to(device)
            text_masks = torch.stack([t2i[3], mmu[3]]).to(device)
            image_masks = torch.stack([t2i[4], mmu[4]]).to(device)
            image_masks[1].zero_()

            t_gen, noise_gen, _ = transport.sample(clean_latent)
            t_gen, latent_gen, label_gen = transport.path_sampler.plan(t_gen, noise_gen, clean_latent)
            t_mmu, noise_mmu, _ = transport.sample(clean_latent, max_t0=1.0)
            t_mmu, latent_mmu, label_mmu = transport.path_sampler.plan(t_mmu, noise_mmu, clean_latent)
            image_latents = torch.cat([latent_gen, latent_mmu], dim=0)
            image_labels = torch.cat([label_gen, label_mmu], dim=0)
            timesteps = torch.cat([t_gen, t_mmu], dim=0)
            attention_mask = omni_attn_mask_naive(
                text_tokens.size(0), text_tokens.size(1), modality_positions, device
            ).to(weight_type)

            torch.cuda.reset_peak_memory_stats(device)
            _, loss_mmu, loss_t2i = model(
                text_tokens=text_tokens,
                image_latents=image_latents,
                t=timesteps.to(weight_type),
                attention_mask=attention_mask,
                text_masks=text_masks,
                image_masks=image_masks,
                text_labels=text_labels,
                image_labels=image_labels,
                modality_positions=modality_positions,
                output_hidden_states=True,
                max_seq_len=max_seq_len,
                device=device,
            )
            grads_mmu = torch.autograd.grad(loss_mmu, params, retain_graph=True, allow_unused=True)
            grads_t2i = torch.autograd.grad(loss_t2i, params, retain_graph=False, allow_unused=True)

            lm_ids = [token_ids["bos_id"], *mmu_ids, token_ids["eos_id"]]
            lm_tokens = torch.tensor(lm_ids, dtype=torch.long, device=device).unsqueeze(0)
            lm_attention = torch.ones_like(lm_tokens, dtype=torch.bool)
            lm_outputs = model.showo(input_ids=lm_tokens, attention_mask=lm_attention, use_cache=False)
            loss_lm = next_token_prediction(lm_outputs.logits, lm_tokens, model.config.llm_vocab_size)
            grads_lm = torch.autograd.grad(loss_lm, params, retain_graph=False, allow_unused=True)

            pairs = {
                "mmu_t2i": pair_metrics(grads_mmu, grads_t2i, group_indices, scopes, params),
                "lm_mmu": pair_metrics(grads_lm, grads_mmu, group_indices, scopes, params),
                "lm_t2i": pair_metrics(grads_lm, grads_t2i, group_indices, scopes, params),
            }
            result = {
                "sample_id": row["sample_id"],
                "index": row.get("index", row_number),
                "tag": row["tag"],
                "correct": row["correct"],
                "image": row["image"],
                "prompt": prompt,
                "sample_seed": sample_seed,
                "timestep_t2i": float(t_gen.item()),
                "losses": {
                    "lm": float(loss_lm.detach().float().item()),
                    "mmu": float(loss_mmu.detach().float().item()),
                    "t2i": float(loss_t2i.detach().float().item()),
                },
                "pairs": pairs,
                "runtime_seconds": time.time() - sample_started,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            }
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            completed.add(row["sample_id"])
            status.update({
                "completed": len(completed),
                "current_sample": None,
                "updated_at": int(time.time()),
                "last_runtime_seconds": result["runtime_seconds"],
            })
            write_json_atomic(status_path, status)
            print(json.dumps({
                "completed": len(completed),
                "total": len(rows),
                "sample_id": row["sample_id"],
                "tag": row["tag"],
                "mmu_t2i_fusion_cosine": pairs["mmu_t2i"]["fusion"]["cosine"],
                "mmu_t2i_qwen_cosine": pairs["mmu_t2i"]["qwen.all"]["cosine"],
                "runtime_seconds": result["runtime_seconds"],
            }), flush=True)
            del grads_mmu, grads_t2i, grads_lm, lm_outputs, loss_lm, loss_mmu, loss_t2i
            gc.collect()
            torch.cuda.empty_cache()
    except BaseException as exc:
        status.update({"status": "failed", "error": repr(exc), "updated_at": int(time.time())})
        write_json_atomic(status_path, status)
        raise
    status.update({
        "status": "completed",
        "completed": len(completed),
        "current_sample": None,
        "updated_at": int(time.time()),
        "finished_at": int(time.time()),
    })
    write_json_atomic(status_path, status)


if __name__ == "__main__":
    main()
