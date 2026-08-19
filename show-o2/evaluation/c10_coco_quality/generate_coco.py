#!/usr/bin/env python3
"""Generate a resumable paired COCO image set with Show-o2."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image

SHOWO2_ROOT = Path(__file__).resolve().parents[2]
if str(SHOWO2_ROOT) not in sys.path:
    sys.path.insert(0, str(SHOWO2_ROOT))

from common import (  # noqa: E402
    ensure_protocol,
    file_sha256,
    generated_image_path,
    read_coco_manifest,
    utc_now,
    verify_image,
    write_json,
)
from evaluation.c8_self_cycle.run_self_cycle import (  # noqa: E402
    checkpoint_construction_kwargs,
    checkpoint_descriptor,
    encode_drsa_prompt_codes,
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
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def dtype_from_name(value: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[value]


def valid_existing(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        info = verify_image(path)
    except (OSError, RuntimeError, ValueError):
        return False
    return (
        int(info["width"]) == 432
        and int(info["height"]) == 432
        and str(info["mode"]) == "RGB"
    )


def wait_for_gate(run_dir: Path, tracker: "Tracker") -> None:
    """Wait for an optional external prerequisite before loading GPU models."""
    config_path = run_dir / "WAIT_FOR_GATE.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gate_path = Path(config["gate_path"]).expanduser().resolve()
    failure_value = config.get("failure_path")
    failure_path = (
        Path(failure_value).expanduser().resolve() if failure_value else None
    )
    poll_seconds = float(config.get("poll_seconds", 15.0))
    if poll_seconds <= 0:
        raise ValueError("WAIT_FOR_GATE poll_seconds must be positive")
    description = str(config.get("description", gate_path))
    while not gate_path.is_file():
        if failure_path is not None and failure_path.is_file():
            raise RuntimeError(
                f"generation prerequisite failed: {description}; marker={failure_path}"
            )
        tracker.update(
            status="running",
            phase="waiting_for_gate",
            gate_description=description,
            gate_path=str(gate_path),
            gate_failure_path=str(failure_path) if failure_path else None,
        )
        time.sleep(poll_seconds)
    tracker.update(
        status="running",
        phase="gate_satisfied",
        gate_description=description,
        gate_path=str(gate_path),
        gate_satisfied_at=utc_now(),
    )


class Tracker:
    def __init__(self, path: Path, variant: str, total: int, sampler_steps: int):
        self.path = path
        if path.is_file():
            self.payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.payload = {
                "variant": variant,
                "started_at": utc_now(),
                "status": "initializing",
            }
        self.payload.update(total_images=total, sampler_steps=sampler_steps)
        self.update()

    def update(self, **values: Any) -> None:
        self.payload.update(values)
        completed = int(self.payload.get("completed_images", 0))
        active = int(self.payload.get("active_images", 0))
        step = int(self.payload.get("sampler_step", 0))
        steps = int(self.payload["sampler_steps"])
        partial = active * step / steps if steps else 0.0
        progress = min(float(self.payload["total_images"]), completed + partial)
        total = int(self.payload["total_images"])
        self.payload["percent"] = 100.0 * progress / total if total else 100.0
        self.payload["updated_at"] = utc_now()
        try:
            started = __import__("datetime").datetime.fromisoformat(
                str(self.payload["started_at"])
            )
            now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
            elapsed = max(0.0, (now - started).total_seconds())
        except (KeyError, ValueError):
            elapsed = 0.0
        self.payload["elapsed_seconds"] = elapsed
        if progress > 0 and progress < total and elapsed > 0:
            self.payload["eta_seconds"] = elapsed * (total - progress) / progress
        elif progress >= total:
            self.payload["eta_seconds"] = 0.0
        write_json(self.path, self.payload)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_inference_steps < 2 or args.generation_batch_size < 1:
        raise ValueError("inference steps must be >=2 and batch size must be positive")
    for path in (args.manifest, args.config, args.vae_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = read_coco_manifest(args.manifest)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[: args.limit]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.run_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.drsa_checkpoint.resolve() if args.drsa_checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_info = checkpoint_descriptor(checkpoint)
    protocol = {
        "protocol_version": "showo2-paired-coco-generation-v2",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "manifest_rows": len(rows),
        "variant_name": args.variant_name,
        "model": args.model,
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "vae_path": str(args.vae_path.resolve()),
        "vae_sha256": file_sha256(args.vae_path),
        "drsa_checkpoint": (
            {"path": str(checkpoint), "sha256": file_sha256(checkpoint)}
            if checkpoint
            else None
        ),
        "weight_type": args.weight_type,
        "vae_weight_type": args.vae_weight_type,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "generation_batch_size": args.generation_batch_size,
        "seed_source": "per-row seed frozen in COCO manifest",
        "conditioning": (
            "legacy prompt relation code"
            if checkpoint_info and checkpoint_info["requires_relation_code"]
            else "generation-only adaptation"
            if checkpoint_info else "native baseline"
        ),
        "checkpoint_kind": checkpoint_info["kind"] if checkpoint_info else None,
        "generation_adaptation": (
            checkpoint_info["route"]
            if checkpoint_info and checkpoint_info["kind"] == "generation_adaptation"
            else None
        ),
        "external_relation_labels": False,
        "output_format": "RGB PNG 432x432",
    }
    ensure_protocol(args.run_dir / "protocol.json", protocol)
    tasks = [(index, row) for index, row in enumerate(rows)]
    pending = [
        task
        for task in tasks
        if not valid_existing(generated_image_path(image_dir, task[0]))
    ]
    tracker = Tracker(
        args.run_dir / "status.json",
        args.variant_name,
        len(tasks),
        args.num_inference_steps - 1,
    )
    tracker.update(
        status="completed" if not pending or args.dry_run else "running",
        phase="dry_run" if args.dry_run else ("completed" if not pending else "validated"),
        completed_images=len(tasks) - len(pending),
        active_images=0,
        sampler_step=0,
    )
    if args.dry_run or not pending:
        summary = {
            "status": "completed",
            "dry_run": args.dry_run,
            "variant": args.variant_name,
            "images": len(tasks) - len(pending),
            "pending_images": len(pending),
            "protocol": protocol,
        }
        write_json(args.run_dir / "generation_summary.json", summary)
        return summary
    wait_for_gate(args.run_dir, tracker)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Show-o2 COCO generation requires CUDA")

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
    seed_everything(int(rows[0]["seed"]))
    tracker.update(phase="loading_model")
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
    model = (
        Showo2Qwen2_5.from_pretrained(
            args.model,
            use_safetensors=False,
            local_files_only=True,
            low_cpu_mem_usage=False,
            **checkpoint_construction_kwargs(checkpoint),
        )
        .to(device=device, dtype=weight_type)
        .eval()
        .requires_grad_(False)
    )
    checkpoint_metadata = load_drsa_checkpoint(model, checkpoint)
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
    started = time.perf_counter()
    try:
        for start in range(0, len(pending), args.generation_batch_size):
            chunk = pending[start : start + args.generation_batch_size]
            prompts = [str(row["prompt"]) for _, row in chunk]
            seeds = [int(row["seed"]) for _, row in chunk]
            tracker.update(
                phase="generation",
                active_images=len(chunk),
                sampler_step=0,
                current_indices=[index for index, _ in chunk],
                current_seeds=seeds,
            )
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
            if args.guidance_scale > 0:
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
                    model,
                    tokenizer,
                    prompts,
                    cfg=args.guidance_scale > 0,
                )

            def progress(record: dict[str, Any]) -> None:
                tracker.update(sampler_step=int(record["step"]) + 1)

            final_latents, _diagnostics = sampler.sample(
                initial_latents=latents,
                model=model,
                model_kwargs=model_kwargs,
                guider=None,
                progress=progress,
            )
            if args.guidance_scale > 0:
                final_latents = torch.chunk(final_latents, 2, dim=0)[0]
            with torch.no_grad():
                decoded = vae.batch_decode(final_latents.unsqueeze(2)).squeeze(2)
            for array, (index, _row) in zip(denorm(decoded), chunk, strict=True):
                save_png_atomic(
                    Image.fromarray(array).convert("RGB"),
                    generated_image_path(image_dir, index),
                )
            completed = sum(
                valid_existing(generated_image_path(image_dir, index))
                for index, _ in tasks
            )
            tracker.update(
                completed_images=completed,
                active_images=0,
                sampler_step=0,
            )
            del latents, final_latents, decoded, attention_mask, model_kwargs
            del text_cond, text_null, text_tokens, modality_positions
            torch.cuda.empty_cache()
    except BaseException as error:
        tracker.update(
            status="failed",
            phase="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise
    completed = sum(
        valid_existing(generated_image_path(image_dir, index)) for index, _ in tasks
    )
    if completed != len(tasks):
        raise RuntimeError(f"generation completed only {completed}/{len(tasks)} images")
    tracker.update(
        status="completed",
        phase="completed",
        completed_images=completed,
        active_images=0,
        sampler_step=0,
    )
    summary = {
        "status": "completed",
        "variant": args.variant_name,
        "images": completed,
        "wall_seconds": time.perf_counter() - started,
        "protocol": protocol,
        "model": checkpoint_metadata,
    }
    write_json(args.run_dir / "generation_summary.json", summary)
    (args.run_dir / "GENERATION_COMPLETED").touch()
    del model, vae, tokenizer, transport
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    try:
        result = run(parse_args())
    except BaseException:
        traceback.print_exc()
        raise
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
