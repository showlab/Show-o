#!/usr/bin/env python3
"""Generate resumable Show-o2 images for an SR2D/VISOR manifest."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image


SHOWO2_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = SHOWO2_ROOT.parents[1]
if str(SHOWO2_ROOT) not in sys.path:
    sys.path.insert(0, str(SHOWO2_ROOT))

from evaluation.c8_self_cycle.run_self_cycle import (  # noqa: E402
    checkpoint_construction_kwargs,
    checkpoint_descriptor,
    encode_drsa_prompt_codes,
    file_sha256,
    load_drsa_checkpoint,
    save_png_atomic,
    seeded_latents,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--drsa-checkpoint", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=SHOWO2_ROOT / "configs/showo2_1.5b_demo_432x432.yaml",
    )
    parser.add_argument("--model", default="showlab/show-o2-1.5B")
    parser.add_argument(
        "--vae-path", type=Path, default=SHOWO2_ROOT / "Wan2.1_VAE.pth"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--weight-type", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--vae-weight-type", choices=("bfloat16", "float32"), default="float32"
    )
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=20260811)
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def ensure_protocol(path: Path, protocol: dict[str, Any]) -> bool:
    normalized = json.loads(json.dumps(protocol, ensure_ascii=True, sort_keys=True))
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise RuntimeError(f"refusing to resume {path.parent}: protocol differs")
        return True
    write_json(path, normalized)
    return False


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("VISOR manifest is empty")
    required = {"unique_id", "prompt", "object_a", "object_b", "relation"}
    for index, row in enumerate(rows):
        if not required.issubset(row):
            raise ValueError(f"VISOR manifest row {index} is missing required fields")
    if len({int(row["unique_id"]) for row in rows}) != len(rows):
        raise ValueError("VISOR manifest unique_id values must be unique")
    return rows


def image_path(run_dir: Path, unique_id: int, sample_index: int) -> Path:
    return run_dir / "images" / f"{unique_id}_{sample_index}.png"


def sample_seed(base_seed: int, unique_id: int, sample_index: int) -> int:
    if sample_index < 0 or sample_index >= 100:
        raise ValueError("VISOR sample indices must be in [0, 100)")
    return base_seed + unique_id * 100 + sample_index


def dtype_from_name(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.samples_per_prompt < 1 or args.samples_per_prompt >= 100:
        raise ValueError("samples-per-prompt must be between 1 and 99")
    if args.generation_batch_size < 1 or args.num_inference_steps < 2:
        raise ValueError("batch size must be positive and inference steps at least two")
    if not args.config.is_file() or not args.vae_path.is_file():
        raise FileNotFoundError("Show-o2 config or VAE is missing")
    records = read_manifest(args.manifest)
    if args.limit_prompts is not None:
        if args.limit_prompts < 1:
            raise ValueError("limit-prompts must be positive")
        records = records[: args.limit_prompts]

    tasks = [
        (record, sample_index)
        for record in records
        for sample_index in range(args.samples_per_prompt)
    ]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_info = checkpoint_descriptor(args.drsa_checkpoint)
    protocol = {
        "protocol_version": "showo2-sr2d-visor-generation-v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "variant_name": args.variant_name,
        "model": args.model,
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "vae_path": str(args.vae_path.resolve()),
        "drsa_checkpoint": (
            {
                "path": str(args.drsa_checkpoint.resolve()),
                "sha256": file_sha256(args.drsa_checkpoint),
            }
            if args.drsa_checkpoint
            else None
        ),
        "checkpoint_kind": checkpoint_info["kind"] if checkpoint_info else None,
        "generation_adaptation": (
            checkpoint_info["route"]
            if checkpoint_info and checkpoint_info["kind"] == "generation_adaptation"
            else None
        ),
        "samples_per_prompt": args.samples_per_prompt,
        "generation_batch_size": args.generation_batch_size,
        "base_seed": args.base_seed,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "limit_prompts": args.limit_prompts,
        "seed_formula": "base_seed + official_unique_id * 100 + sample_index",
        "external_relation_labels": False,
        "conditioning": (
            "legacy prompt relation code"
            if checkpoint_info and checkpoint_info["requires_relation_code"]
            else "generation-only adaptation"
            if checkpoint_info else "native baseline"
        ),
    }
    ensure_protocol(args.run_dir / "protocol.json", protocol)
    pending = [
        (record, sample_index)
        for record, sample_index in tasks
        if not image_path(args.run_dir, int(record["unique_id"]), sample_index).is_file()
    ]
    status = {
        "status": "completed" if args.dry_run or not pending else "running",
        "phase": "dry_run" if args.dry_run else "generation",
        "variant_name": args.variant_name,
        "prompts": len(records),
        "total_images": len(tasks),
        "completed_images": len(tasks) - len(pending),
        "started_at": utc_now(),
        "updated_at": utc_now(),
    }
    write_json(args.run_dir / "status.json", status)
    if args.dry_run or not pending:
        status.update(status="completed", phase="completed")
        write_json(args.run_dir / "status.json", status)
        return {"protocol": protocol, **status}
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Show-o2 VISOR generation requires CUDA")

    from omegaconf import OmegaConf

    from guidance.sampler import GuidedEulerSampler
    from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
    from models.misc import get_text_tokenizer, prepare_gen_input
    from transport import create_transport
    from utils import denorm, get_hyper_params, path_to_llm_name

    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)
    device = torch.device(args.device)
    weight_type = dtype_from_name(args.weight_type)
    vae_weight_type = dtype_from_name(args.vae_weight_type)
    seed_everything(args.base_seed)
    tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
    )
    config.model.showo.llm_vocab_size = len(tokenizer)
    if bool(config.model.showo.add_time_embeds):
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
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
        _default_guidance_scale,
    ) = get_hyper_params(config, tokenizer, showo_token_ids)
    checkpoint_kwargs = checkpoint_construction_kwargs(args.drsa_checkpoint)
    model = (
        Showo2Qwen2_5.from_pretrained(
            args.model,
            use_safetensors=False,
            local_files_only=True,
            low_cpu_mem_usage=False,
            **checkpoint_kwargs,
        )
        .to(device=device, dtype=weight_type)
        .eval()
        .requires_grad_(False)
    )
    checkpoint_metadata = load_drsa_checkpoint(model, args.drsa_checkpoint)
    write_json(args.run_dir / "model_metadata.json", checkpoint_metadata)
    vae = WanVAE(vae_pth=str(args.vae_path), dtype=vae_weight_type, device=device)
    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=num_t2i_image_tokens,
    )
    sampler = GuidedEulerSampler(
        transport=transport,
        num_steps=args.num_inference_steps,
        guidance_steps=0,
        time_shifting_factor=(
            float(config.transport.time_shifting_factor)
            if config.transport.time_shifting_factor is not None
            else None
        ),
        reverse=bool(config.transport.reverse),
    )
    latent_shape = (
        1,
        image_latent_dim,
        latent_height * patch_size,
        latent_width * patch_size,
    )

    for start in range(0, len(pending), args.generation_batch_size):
        chunk = pending[start : start + args.generation_batch_size]
        prompts = [str(record["prompt"]) for record, _ in chunk]
        seeds = [
            sample_seed(args.base_seed, int(record["unique_id"]), sample_index)
            for record, sample_index in chunk
        ]
        text_cond, text_null, positions_cond, positions_null = prepare_gen_input(
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
        latents = seeded_latents(seeds, latent_shape, device, weight_type)
        if args.guidance_scale > 0.0:
            latents = torch.cat([latents, latents], dim=0)
            text_tokens = torch.cat([text_cond, text_null], dim=0)
            modality_positions = torch.cat([positions_cond, positions_null], dim=0)
        else:
            text_tokens = text_cond
            modality_positions = positions_cond
        attention_mask = omni_attn_mask_naive(
            text_tokens.size(0), max_seq_len, modality_positions, device
        ).to(weight_type)
        model_kwargs = {
            "text_tokens": text_tokens,
            "attention_mask": attention_mask,
            "modality_positions": modality_positions,
            "output_hidden_states": True,
            "max_seq_len": max_seq_len,
            "guidance_scale": args.guidance_scale,
        }
        if checkpoint_metadata.get("requires_relation_code", False):
            model_kwargs["drsa_relation_codes"] = encode_drsa_prompt_codes(
                model, tokenizer, prompts, cfg=args.guidance_scale > 0.0
            )
        final_latents, _diagnostics = sampler.sample(
            initial_latents=latents,
            model=model,
            model_kwargs=model_kwargs,
            guider=None,
            progress=None,
        )
        if args.guidance_scale > 0.0:
            final_latents = torch.chunk(final_latents, 2, dim=0)[0]
        with torch.no_grad():
            decoded = vae.batch_decode(final_latents.unsqueeze(2)).squeeze(2)
        arrays = denorm(decoded)
        for array, (record, sample_index) in zip(arrays, chunk):
            path = image_path(args.run_dir, int(record["unique_id"]), sample_index)
            save_png_atomic(Image.fromarray(array), path)
        completed = len(tasks) - sum(
            not image_path(args.run_dir, int(record["unique_id"]), sample_index).is_file()
            for record, sample_index in tasks
        )
        status.update(completed_images=completed, updated_at=utc_now())
        write_json(args.run_dir / "status.json", status)
        del latents, final_latents, decoded, attention_mask, model_kwargs
        del text_cond, text_null, text_tokens, modality_positions
        torch.cuda.empty_cache()

    status.update(status="completed", phase="completed", updated_at=utc_now())
    write_json(args.run_dir / "status.json", status)
    summary = {"status": "completed", "protocol": protocol, "model": checkpoint_metadata}
    write_json(args.run_dir / "generation_summary.json", summary)
    del model, vae, transport, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
