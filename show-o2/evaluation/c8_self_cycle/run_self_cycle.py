#!/usr/bin/env python3
"""Run a resumable Show-o2 text-to-image-to-structured-QA benchmark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

SHOWO2_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = SHOWO2_ROOT.parents[1]
if str(SHOWO2_ROOT) not in sys.path:
    sys.path.insert(0, str(SHOWO2_ROOT))

from evaluation.c8_self_cycle.scoring import (  # noqa: E402
    QuestionSpec,
    aggregate_results,
    classify_question,
    score_answer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(WORKSPACE_ROOT / "data/showo2_self_cycle/unicycle/self_built.jsonl"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SHOWO2_ROOT / "configs/showo2_1.5b_demo_432x432.yaml",
    )
    parser.add_argument("--model", default="showlab/show-o2-1.5B")
    parser.add_argument("--variant-name", default="baseline")
    parser.add_argument("--drsa-checkpoint", type=Path)
    parser.add_argument(
        "--vae-path",
        type=Path,
        default=SHOWO2_ROOT / "Wan2.1_VAE.pth",
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
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260809)
    parser.add_argument("--max-open-answer-tokens", type=int, default=12)
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument(
        "--record-indices", help="Comma-separated dataset indices for smoke runs"
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dtype_from_name(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        rows.append(row)
    return rows


def prompt_from_record(record: dict[str, Any]) -> str:
    value = record.get("short_description", record.get("short_desription", ""))
    prompt = str(value).strip()
    if not prompt:
        raise ValueError("record has no short_description")
    return prompt


def validate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("self-cycle dataset must not be empty")
    type_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    questions = 0
    for record_index, record in enumerate(records):
        prompt_from_record(record)
        qa_pairs = record.get("qa_pairs")
        if not isinstance(qa_pairs, list) or not qa_pairs:
            raise ValueError(f"record {record_index} has no qa_pairs")
        type_counts[str(record.get("type", "unknown"))] += 1
        for qa_index, qa in enumerate(qa_pairs):
            if not isinstance(qa, dict):
                raise TypeError(f"record {record_index} qa {qa_index} is not an object")
            question = str(qa.get("question", "")).strip()
            answer = qa.get("answer")
            if not question or answer is None or not str(answer).strip():
                raise ValueError(f"record {record_index} qa {qa_index} is incomplete")
            mode_counts[classify_question(question, answer).mode] += 1
            questions += 1
    return {
        "records": len(records),
        "questions": questions,
        "type_counts": dict(sorted(type_counts.items())),
        "answer_mode_counts": dict(sorted(mode_counts.items())),
    }


def ensure_protocol(path: Path, protocol: dict[str, Any]) -> bool:
    normalized = json.loads(json.dumps(protocol, ensure_ascii=True, sort_keys=True))
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise RuntimeError(
                f"Refusing to resume {path.parent}: protocol.json differs"
            )
        return True
    write_json(path, normalized)
    return False


class StatusTracker:
    def __init__(
        self,
        path: Path,
        run_dir: Path,
        total_images: int,
        total_questions: int,
        resume: bool,
    ):
        self.path = path
        if resume and path.is_file():
            self.payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.payload = {
                "run_dir": str(run_dir.resolve()),
                "status": "initializing",
                "phase": "initializing",
                "started_at": utc_now(),
                "completed_images": 0,
                "completed_questions": 0,
            }
        self.payload["total_images"] = total_images
        self.payload["total_questions"] = total_questions
        self.update()

    def update(self, **values: Any) -> None:
        if values.get("status") != "failed":
            self.payload.pop("error", None)
            self.payload.pop("traceback", None)
        self.payload.update(values)
        completed_images = int(self.payload.get("completed_images", 0))
        completed_questions = int(self.payload.get("completed_questions", 0))
        total_images = int(self.payload["total_images"])
        total_questions = int(self.payload["total_questions"])
        active_images = int(self.payload.get("active_generation_images", 0))
        sampler_step = int(self.payload.get("sampler_step", 0))
        sampler_steps = int(self.payload.get("sampler_steps", 0))
        partial_images = (
            active_images * sampler_step / sampler_steps if sampler_steps else 0.0
        )
        generation_progress = min(total_images, completed_images + partial_images)
        total_units = total_images + total_questions
        completed_units = min(total_units, generation_progress + completed_questions)
        self.payload["generation_percent"] = (
            100.0 * generation_progress / total_images if total_images else 100.0
        )
        self.payload["qa_percent"] = (
            100.0 * completed_questions / total_questions if total_questions else 100.0
        )
        self.payload["percent"] = (
            100.0 * completed_units / total_units if total_units else 100.0
        )
        try:
            started = datetime.fromisoformat(str(self.payload["started_at"]))
            elapsed = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
        except (KeyError, ValueError):
            elapsed = 0.0
        self.payload["elapsed_seconds"] = elapsed
        if completed_units > 0.0 and completed_units < total_units and elapsed > 0.0:
            rate = completed_units / elapsed
            self.payload["work_units_per_second"] = rate
            self.payload["eta_seconds"] = (total_units - completed_units) / rate
        elif completed_units >= total_units:
            self.payload["eta_seconds"] = 0.0
        self.payload["updated_at"] = utc_now()
        write_json(self.path, self.payload)


def result_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["record_index"]),
        int(row["sample_index"]),
        int(row["qa_index"]),
    )


def load_results(path: Path) -> dict[tuple[int, int, int], dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[tuple[int, int, int], dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        key = result_key(row)
        if key in rows:
            raise ValueError(f"duplicate result at {path}:{line_number}: {key}")
        rows[key] = row
    return rows


def sample_seed(base_seed: int, record_index: int, sample_index: int) -> int:
    if sample_index >= 100:
        raise ValueError("samples_per_prompt must be below 100")
    return base_seed + record_index * 100 + sample_index


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seeded_latents(
    seeds: list[int],
    latent_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        values.append(
            torch.randn(
                latent_shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        )
    return torch.cat(values, dim=0)


def image_path(run_dir: Path, record_index: int, sample_index: int) -> Path:
    return run_dir / "images" / f"{record_index:05d}" / f"sample_{sample_index:03d}.png"


def save_png_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.png")
    image.save(temporary)
    os.replace(temporary, path)


def checkpoint_construction_kwargs(checkpoint_path: Path | None) -> dict[str, Any]:
    if checkpoint_path is None:
        return {"drsa_enabled": False, "generation_adapter_enabled": False}
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    from models.generation_adaptation import checkpoint_model_kwargs

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    kwargs = checkpoint_model_kwargs(payload)
    del payload
    return kwargs


def checkpoint_descriptor(checkpoint_path: Path | None) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "adaptation_config" in payload:
        descriptor = {
            "kind": "generation_adaptation",
            "route": dict(payload["adaptation_config"]),
            "requires_relation_code": False,
        }
    elif "drsa_adapter" in payload:
        descriptor = {
            "kind": "legacy_drsa",
            "route": None,
            "requires_relation_code": True,
        }
    else:
        raise ValueError(
            "checkpoint contains neither adaptation_state nor drsa_adapter"
        )
    descriptor.update({
        "path": str(checkpoint_path.resolve()),
        "sha256": file_sha256(checkpoint_path),
        "step": int(payload.get("step", 0)),
    })
    del payload
    return descriptor


def load_drsa_checkpoint(model: Any, checkpoint_path: Path | None) -> dict[str, Any]:
    """Load legacy DRSA or the new generation-only adaptation checkpoint."""
    if checkpoint_path is None:
        adaptation = getattr(model, "_generation_adaptation", None)
        if adaptation is not None:
            adaptation.set_enabled(False)
        return {"enabled": False, "checkpoint": None}
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "adaptation_config" in payload:
        from models.generation_adaptation import configure_from_checkpoint

        adaptation = getattr(model, "_generation_adaptation", None)
        route = str(payload["adaptation_config"]["route"])
        if adaptation is None:
            adaptation = configure_from_checkpoint(model, payload)
            model._generation_adaptation = adaptation
        else:
            if adaptation.route != route:
                raise RuntimeError(
                    f"cannot switch generation adaptation from {adaptation.route} to {route}"
                )
            adaptation.load_state_dict(payload["adaptation_state"], strict=True)
        adaptation.requires_grad_(False)
        adaptation.train(False)
        adaptation.set_enabled(True)
        parameter_norm = math.sqrt(sum(
            float(value.detach().float().square().sum())
            for value in adaptation.state_dict().values()
        ))
        kind = "generation_adaptation"
        requires_relation_code = False
        route_config = adaptation.config_dict()
    else:
        if model.drsa_adapter is None:
            raise RuntimeError("DRSA checkpoint requested but model has no adapter")
        state = payload["drsa_adapter"]
        if any(key.startswith("generation_spatial.") for key in state):
            model.drsa_adapter.enable_generation_spatial()
        model.drsa_adapter.load_state_dict(state, strict=True)
        parameter_norm = math.sqrt(sum(
            float(value.detach().float().square().sum())
            for value in model.drsa_adapter.state_dict().values()
        ))
        kind = "legacy_drsa"
        requires_relation_code = True
        route_config = None
    metadata = {
        "enabled": True,
        "kind": kind,
        "route": route_config,
        "requires_relation_code": requires_relation_code,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "step": int(payload["step"]),
        "parameter_l2": parameter_norm,
    }
    del payload
    return metadata


def encode_drsa_prompt_codes(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    cfg: bool,
) -> torch.Tensor:
    if model.drsa_adapter is None:
        raise RuntimeError("DRSA prompt encoding requires an initialized adapter")
    if not prompts:
        raise ValueError("DRSA prompt encoding requires at least one prompt")
    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    with torch.no_grad():
        features = model.showo.model.embed_tokens(token_ids)
        codes = model.drsa_adapter.encode_relation(features, attention_mask)
    if codes.ndim != 2 or codes.shape[0] != len(prompts):
        raise RuntimeError(
            f"invalid DRSA prompt-code shape {tuple(codes.shape)} for "
            f"{len(prompts)} prompts"
        )
    if not torch.isfinite(codes).all():
        raise RuntimeError("DRSA prompt codes contain non-finite values")
    if cfg:
        codes = torch.cat([codes, codes], dim=0)
    return codes


def extend_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 4 or mask.shape[:2] != (1, 1):
        raise ValueError(f"expected [1,1,L,L] mask, got {tuple(mask.shape)}")
    length = mask.shape[-1]
    minimum = torch.finfo(mask.dtype).min
    result = torch.full(
        (1, 1, length + 1, length + 1),
        minimum,
        device=mask.device,
        dtype=mask.dtype,
    )
    result[:, :, :length, :length] = mask
    result[:, :, length, :length] = mask[:, :, length - 1, :]
    result[:, :, length, length] = 0
    return result


def mmu_question_prompt(question: str, mode: str) -> str:
    if mode == "multiple_choice":
        instruction = "Answer with exactly one option letter: A, B, C, or D."
    elif mode == "yes_no":
        instruction = "Answer with exactly one lowercase word: yes or no."
    else:
        instruction = "Answer with one word or a short phrase. Do not explain."
    return (
        "Inspect only the provided image. Do not infer from an earlier text prompt. "
        f"{instruction}\nQuestion: {question.strip()}\nAnswer:"
    )


def build_mmu_prefix(
    model: Any,
    tokenizer: Any,
    showo_token_ids: dict[str, int],
    image_embeds: torch.Tensor,
    question_prompt: str,
    num_mmu_image_tokens: int,
    add_time_embeds: bool,
    weight_type: torch.dtype,
    device: torch.device,
    omni_attn_mask_naive: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    system_ids = tokenizer(
        "system\nYou are a helpful assistant.<|im_end|>",
        add_special_tokens=False,
    )["input_ids"]
    role_user = tokenizer("\n<|im_start|>user\n", add_special_tokens=False)["input_ids"]
    role_assistant = tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False)[
        "input_ids"
    ]
    question_ids = tokenizer(question_prompt, add_special_tokens=False)["input_ids"]
    tokens_a = torch.tensor(
        [showo_token_ids["bos_id"]] + system_ids + role_user,
        device=device,
    )[None, :]
    tokens_b = torch.tensor(
        [showo_token_ids["boi_id"], showo_token_ids["eoi_id"]]
        + question_ids
        + role_assistant,
        device=device,
    )[None, :]
    embeds_a = model.showo.model.embed_tokens(tokens_a)
    embeds_b = model.showo.model.embed_tokens(tokens_b)
    if add_time_embeds:
        time_embeds = model.time_embed(
            torch.tensor([[1.0]], device=device), embeds_a.dtype
        )
        if hasattr(model, "time_embed_proj"):
            time_embeds = model.time_embed_proj(time_embeds)
        input_embeds = torch.cat(
            [embeds_a, embeds_b[:, :1], time_embeds, image_embeds, embeds_b[:, 1:]],
            dim=1,
        ).to(weight_type)
        offset = tokens_a.shape[1] + 2
    else:
        input_embeds = torch.cat(
            [embeds_a, embeds_b[:, :1], image_embeds, embeds_b[:, 1:]], dim=1
        ).to(weight_type)
        offset = tokens_a.shape[1] + 1
    positions = torch.tensor(
        [[[offset, num_mmu_image_tokens]]], device=device, dtype=torch.long
    )
    attention_mask = omni_attn_mask_naive(
        B=1,
        LEN=input_embeds.shape[1],
        modalities=positions,
        device=device,
        inverted=True,
    ).to(weight_type)
    return input_embeds, attention_mask


@torch.no_grad()
def encode_image_for_mmu(
    image: Image.Image,
    model: Any,
    vae: Any,
    resolution: int,
    weight_type: torch.dtype,
    device: torch.device,
    image_transform: Any,
) -> torch.Tensor:
    tensor = image_transform(image.convert("RGB"), resolution=resolution)
    tensor = tensor.to(device).unsqueeze(0)
    latents = vae.sample(tensor.unsqueeze(2), deterministic=True).squeeze(2)
    latents = latents.to(weight_type)
    embeds_und = model.image_embedder_und(latents)
    embeds_gen = model.image_embedder_gen(latents)
    embeds_und = embeds_und + model.position_embedding(model.image_position_ids)
    embeds_und = model.und_trans(embeds_und)["last_hidden_state"]
    return model.fusion_proj(torch.cat([embeds_und, embeds_gen], dim=-1))


@torch.no_grad()
def score_candidates(
    model: Any,
    tokenizer: Any,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    candidates: tuple[str, ...],
) -> tuple[str, dict[str, float]]:
    if not candidates:
        raise ValueError("candidate scoring requires candidates")
    initial_logits = model.showo(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask.to(input_embeds.dtype),
    )["logits"][:, -1, :]
    initial_log_probs = torch.log_softmax(initial_logits.float(), dim=-1)
    scores: dict[str, float] = {}
    for candidate in candidates:
        token_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError(f"candidate has no tokens: {candidate!r}")
        total = initial_log_probs[0, token_ids[0]]
        current_embeds = input_embeds
        current_mask = attention_mask
        previous_token = token_ids[0]
        for token_id in token_ids[1:]:
            token_tensor = torch.tensor(
                [[previous_token]], device=input_embeds.device, dtype=torch.long
            )
            current_embeds = torch.cat(
                [current_embeds, model.showo.model.embed_tokens(token_tensor)], dim=1
            ).to(input_embeds.dtype)
            current_mask = extend_attention_mask(current_mask)
            logits = model.showo(
                inputs_embeds=current_embeds,
                attention_mask=current_mask.to(input_embeds.dtype),
            )["logits"][:, -1, :]
            total = total + torch.log_softmax(logits.float(), dim=-1)[0, token_id]
            previous_token = token_id
        scores[candidate] = float((total / len(token_ids)).cpu())
    prediction = max(scores, key=scores.get)
    return prediction, scores


@torch.no_grad()
def generate_short_answer(
    model: Any,
    tokenizer: Any,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> str:
    output = model.mmu_generate(
        input_embeds=input_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        top_k=None,
        eos_token=tokenizer.eos_token_id,
    )
    token_ids = [int(value.detach().cpu()) for value in output]
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    return decoded.splitlines()[0] if decoded else ""


def build_tasks(
    records: list[dict[str, Any]], samples_per_prompt: int
) -> list[tuple[int, int]]:
    return [
        (record_index, sample_index)
        for record_index in range(len(records))
        for sample_index in range(samples_per_prompt)
    ]


def count_existing_images(run_dir: Path, tasks: list[tuple[int, int]]) -> int:
    return sum(image_path(run_dir, *task).is_file() for task in tasks)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_inference_steps < 2:
        raise ValueError("num_inference_steps must be at least 2")
    if args.generation_batch_size < 1 or args.samples_per_prompt < 1:
        raise ValueError("batch size and samples_per_prompt must be positive")
    if args.max_open_answer_tokens < 1:
        raise ValueError("max_open_answer_tokens must be positive")
    if not args.config.is_file() or not args.vae_path.is_file():
        raise FileNotFoundError("config or VAE path is missing")

    records = read_jsonl(args.dataset)
    selected_record_indices = None
    if args.record_indices:
        if args.limit_prompts is not None:
            raise ValueError("record_indices and limit_prompts are mutually exclusive")
        selected_record_indices = [
            int(value.strip())
            for value in args.record_indices.split(",")
            if value.strip()
        ]
        if not selected_record_indices or len(set(selected_record_indices)) != len(
            selected_record_indices
        ):
            raise ValueError("record_indices must contain unique integers")
        if min(selected_record_indices) < 0 or max(selected_record_indices) >= len(
            records
        ):
            raise IndexError("record_indices contains an out-of-range index")
        records = [records[index] for index in selected_record_indices]
    if args.limit_prompts is not None:
        if args.limit_prompts < 1:
            raise ValueError("limit_prompts must be positive")
        records = records[: args.limit_prompts]
    dataset_stats = validate_records(records)
    total_images = len(records) * args.samples_per_prompt
    total_questions = dataset_stats["questions"] * args.samples_per_prompt
    tasks = build_tasks(records, args.samples_per_prompt)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.run_dir / "question_results.jsonl"
    if args.no_resume and any(args.run_dir.iterdir()):
        raise RuntimeError(f"--no-resume requires an empty run dir: {args.run_dir}")
    checkpoint_metadata = checkpoint_descriptor(args.drsa_checkpoint)
    legacy_drsa = bool(
        checkpoint_metadata and checkpoint_metadata["kind"] == "legacy_drsa"
    )
    generation_adaptation = (
        checkpoint_metadata.get("route")
        if checkpoint_metadata and not legacy_drsa
        else None
    )
    protocol = {
        "protocol_version": (
            "c8-self-cycle-generation-adaptation-v1"
            if generation_adaptation
            else "c8-self-cycle-prompt-conditioned-drsa-v1"
            if legacy_drsa else "c8-self-cycle-structured-v1"
        ),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "dataset_stats": dataset_stats,
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "model": args.model,
        "variant_name": args.variant_name,
        "drsa_checkpoint": checkpoint_metadata,
        "generation_adaptation": generation_adaptation,
        "drsa_conditioning": (
            {
                "external_relation_labels": False,
                "generation": "short_description_prompt_embedding",
                "max_text_tokens": 32,
                "mmu": None,
            }
            if legacy_drsa
            else None
        ),
        "vae_path": str(args.vae_path.resolve()),
        "weight_type": args.weight_type,
        "vae_weight_type": args.vae_weight_type,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "generation_batch_size": args.generation_batch_size,
        "samples_per_prompt": args.samples_per_prompt,
        "base_seed": args.base_seed,
        "max_open_answer_tokens": args.max_open_answer_tokens,
        "limit_prompts": args.limit_prompts,
        "record_indices": selected_record_indices,
        "mmu_image_source": "saved_png_reloaded",
        "mmu_vae_encoding": "posterior_mean_deterministic",
        "closed_answer_scoring": "candidate_log_likelihood",
        "open_answer_scoring": "greedy_exact_normalized_and_token_f1",
        "external_llm_judge": False,
        "generation_prompt_hidden_from_mmu": True,
    }
    resume = ensure_protocol(args.run_dir / "protocol.json", protocol)
    completed_results = load_results(results_path) if resume else {}
    tracker = StatusTracker(
        args.run_dir / "status.json",
        args.run_dir,
        total_images,
        total_questions,
        resume=resume,
    )
    tracker.update(
        status="running",
        phase="validated",
        completed_images=count_existing_images(args.run_dir, tasks),
        completed_questions=len(completed_results),
        active_generation_images=0,
        sampler_step=0,
        sampler_steps=args.num_inference_steps - 1,
    )
    write_json(args.run_dir / "dataset_stats.json", dataset_stats)
    if args.dry_run:
        summary = {
            "status": "completed",
            "dry_run": True,
            "protocol": protocol,
            "dataset_stats": dataset_stats,
        }
        write_json(args.run_dir / "summary.json", summary)
        tracker.update(status="completed", phase="dry_run_completed")
        return summary

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("self-cycle benchmark requires CUDA")

    from omegaconf import OmegaConf

    from datasets.utils import image_transform
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
    add_time_embeds = bool(config.model.showo.add_time_embeds)
    if add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1

    (
        num_t2i_image_tokens,
        num_mmu_image_tokens,
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

    tracker.update(phase="loading_model")
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
    adapter_metadata = load_drsa_checkpoint(model, args.drsa_checkpoint)
    write_json(args.run_dir / "model_metadata.json", adapter_metadata)
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
    time_shifting_factor = (
        float(config.transport.time_shifting_factor)
        if config.transport.time_shifting_factor is not None
        else None
    )
    latent_shape = (
        1,
        image_latent_dim,
        latent_height * patch_size,
        latent_width * patch_size,
    )

    pending_generation = [
        task for task in tasks if not image_path(args.run_dir, *task).is_file()
    ]
    sampler = GuidedEulerSampler(
        transport=transport,
        num_steps=args.num_inference_steps,
        guidance_steps=0,
        time_shifting_factor=time_shifting_factor,
        reverse=bool(config.transport.reverse),
    )
    for start in range(0, len(pending_generation), args.generation_batch_size):
        chunk = pending_generation[start : start + args.generation_batch_size]
        prompts = [
            prompt_from_record(records[record_index]) for record_index, _ in chunk
        ]
        seeds = [
            sample_seed(args.base_seed, record_index, sample_index)
            for record_index, sample_index in chunk
        ]
        tracker.update(
            phase="generation",
            active_generation_images=len(chunk),
            sampler_step=0,
            current_record_indices=[item[0] for item in chunk],
            current_sample_indices=[item[1] for item in chunk],
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
        if adapter_metadata.get("requires_relation_code", False):
            relation_codes = encode_drsa_prompt_codes(
                model,
                tokenizer,
                prompts,
                cfg=args.guidance_scale > 0.0,
            )
            model_kwargs["drsa_relation_codes"] = relation_codes
            if start == 0:
                print(
                    json.dumps(
                        {
                            "checkpoint_step": adapter_metadata["step"],
                            "conditioning": "short_description_prompt_embedding",
                            "external_relation_labels": False,
                            "relation_code_l2_mean": float(
                                relation_codes.float().norm(dim=-1).mean().cpu()
                            ),
                            "relation_code_shape": list(relation_codes.shape),
                            "status": "drsa_active",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        def report(step_record: dict[str, Any]) -> None:
            tracker.update(sampler_step=int(step_record["step"]) + 1)

        final_latents, _diagnostics = sampler.sample(
            initial_latents=latents,
            model=model,
            model_kwargs=model_kwargs,
            guider=None,
            progress=report,
        )
        if args.guidance_scale > 0.0:
            final_latents = torch.chunk(final_latents, 2, dim=0)[0]
        with torch.no_grad():
            decoded = vae.batch_decode(final_latents.unsqueeze(2)).squeeze(2)
        arrays = denorm(decoded)
        for array, task in zip(arrays, chunk):
            save_png_atomic(Image.fromarray(array), image_path(args.run_dir, *task))
        tracker.update(
            completed_images=count_existing_images(args.run_dir, tasks),
            active_generation_images=0,
            sampler_step=0,
        )
        del latents, final_latents, decoded, attention_mask, model_kwargs
        del text_cond, text_null, text_tokens, modality_positions
        torch.cuda.empty_cache()

    generation_adaptation = getattr(model, "_generation_adaptation", None)
    if generation_adaptation is not None:
        generation_adaptation.set_enabled(False)
        adapter_metadata["disabled_for_structured_qa"] = True
        write_json(args.run_dir / "model_metadata.json", adapter_metadata)

    tracker.update(
        phase="structured_qa",
        completed_images=count_existing_images(args.run_dir, tasks),
        active_generation_images=0,
        sampler_step=0,
    )
    for record_index, sample_index in tasks:
        record = records[record_index]
        pending_qas = [
            (qa_index, qa)
            for qa_index, qa in enumerate(record["qa_pairs"])
            if (record_index, sample_index, qa_index) not in completed_results
        ]
        if not pending_qas:
            continue
        path = image_path(args.run_dir, record_index, sample_index)
        with Image.open(path) as handle:
            generated_image = handle.convert("RGB")
        image_embeds = encode_image_for_mmu(
            generated_image,
            model,
            vae,
            int(config.dataset.preprocessing.resolution),
            weight_type,
            device,
            image_transform,
        )
        for qa_index, qa in pending_qas:
            question = str(qa["question"]).strip()
            raw_reference = str(qa["answer"]).strip()
            spec: QuestionSpec = classify_question(question, raw_reference)
            prompt = mmu_question_prompt(question, spec.mode)
            input_embeds, qa_mask = build_mmu_prefix(
                model,
                tokenizer,
                showo_token_ids,
                image_embeds,
                prompt,
                num_mmu_image_tokens,
                add_time_embeds,
                weight_type,
                device,
                omni_attn_mask_naive,
            )
            candidate_scores = None
            if spec.candidates:
                prediction, candidate_scores = score_candidates(
                    model, tokenizer, input_embeds, qa_mask, spec.candidates
                )
            else:
                prediction = generate_short_answer(
                    model,
                    tokenizer,
                    input_embeds,
                    qa_mask,
                    args.max_open_answer_tokens,
                )
            scored = score_answer(prediction, spec)
            result = {
                "protocol_version": protocol["protocol_version"],
                "variant_name": args.variant_name,
                "record_index": record_index,
                "sample_index": sample_index,
                "qa_index": qa_index,
                "seed": sample_seed(args.base_seed, record_index, sample_index),
                "type": str(record.get("type", "unknown")),
                "generation_prompt": prompt_from_record(record),
                "generation_prompt_exposed_to_mmu": False,
                "image_path": str(path.resolve()),
                "question": question,
                "reference_answer": raw_reference,
                "answer_mode": spec.mode,
                "candidates": list(spec.candidates),
                "candidate_log_scores": candidate_scores,
                "prediction": prediction,
                **scored,
            }
            append_jsonl(results_path, result)
            completed_results[result_key(result)] = result
            tracker.update(
                phase="structured_qa",
                completed_questions=len(completed_results),
                current_record_index=record_index,
                current_sample_index=sample_index,
                current_qa_index=qa_index,
                current_answer_mode=spec.mode,
            )
            del input_embeds, qa_mask
        del image_embeds, generated_image
        torch.cuda.empty_cache()

    ordered_results = [completed_results[key] for key in sorted(completed_results)]
    if len(ordered_results) != total_questions:
        raise RuntimeError(
            f"expected {total_questions} question results, got {len(ordered_results)}"
        )
    metrics = aggregate_results(ordered_results)
    summary = {
        "status": "completed",
        "completed_at": utc_now(),
        "run_dir": str(args.run_dir.resolve()),
        "variant_name": args.variant_name,
        "dataset": str(args.dataset.resolve()),
        "model": args.model,
        "adapter": adapter_metadata,
        "protocol": protocol,
        "metrics": metrics,
    }
    write_json(args.run_dir / "summary.json", summary)
    tracker.update(
        status="completed",
        phase="completed",
        completed_images=total_images,
        completed_questions=total_questions,
        active_generation_images=0,
        sampler_step=0,
        summary=str((args.run_dir / "summary.json").resolve()),
    )
    del model, vae, transport, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main() -> int:
    args = parse_args()
    tracker_path = args.run_dir / "status.json"
    try:
        summary = run(args)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "failed",
            "phase": "failed",
            "updated_at": utc_now(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        if tracker_path.is_file():
            payload = json.loads(tracker_path.read_text(encoding="utf-8"))
            payload.update(failure)
            write_json(tracker_path, payload)
        else:
            write_json(tracker_path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
