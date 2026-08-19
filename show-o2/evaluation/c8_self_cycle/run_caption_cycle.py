#!/usr/bin/env python3
"""Evaluate a strict Show-o2 text-to-image-to-caption reconstruction cycle."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from collections import defaultdict
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

from evaluation.c8_self_cycle.caption_metrics import lexical_scores  # noqa: E402
from evaluation.c8_self_cycle.run_self_cycle import (  # noqa: E402
    atomic_write_text,
    build_mmu_prefix,
    checkpoint_construction_kwargs,
    dtype_from_name,
    encode_image_for_mmu,
    file_sha256,
    load_drsa_checkpoint,
    prompt_from_record,
    read_jsonl,
    write_json,
)


DEFAULT_GENERATION_RUN = Path(
    "/root/autodl-tmp/runs/showo2_c8/"
    "c8-self-cycle-baseline-20260809T135133Z"
)
DEFAULT_SEMANTIC_MODEL = Path(
    "/root/autodl-tmp/cache/huggingface/hub/"
    "models--sentence-transformers--all-mpnet-base-v2/snapshots/"
    "e8c3b32edf5434bc2275fc9bab85f82640a19130"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--generation-run", type=Path, default=DEFAULT_GENERATION_RUN
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "data/showo2_self_cycle/unicycle/self_built.jsonl"
        ),
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
        "--vae-path", type=Path, default=SHOWO2_ROOT / "Wan2.1_VAE.pth"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--weight-type", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--vae-weight-type", choices=("bfloat16", "float32"), default="float32"
    )
    parser.add_argument("--max-caption-tokens", type=int, default=64)
    parser.add_argument(
        "--semantic-model", type=Path, default=DEFAULT_SEMANTIC_MODEL
    )
    parser.add_argument("--semantic-batch-size", type=int, default=64)
    parser.add_argument("--record-indices")
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    value = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows
    )
    atomic_write_text(path, value)


def select_records(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> list[tuple[int, dict[str, Any]]]:
    if args.record_indices and args.limit_prompts is not None:
        raise ValueError("record_indices and limit_prompts are mutually exclusive")
    if args.record_indices:
        indices = [
            int(value.strip())
            for value in args.record_indices.split(",")
            if value.strip()
        ]
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("record_indices must contain unique integers")
        if min(indices) < 0 or max(indices) >= len(records):
            raise IndexError("record_indices contains an out-of-range value")
    elif args.limit_prompts is not None:
        if args.limit_prompts < 1:
            raise ValueError("limit_prompts must be positive")
        indices = list(range(min(args.limit_prompts, len(records))))
    else:
        indices = list(range(len(records)))
    return [(index, records[index]) for index in indices]


def source_image(generation_run: Path, record_index: int) -> Path:
    return generation_run / "images" / f"{record_index:05d}" / "sample_000.png"


def load_captions(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        index = int(row["record_index"])
        if index in rows:
            raise ValueError(f"duplicate caption at {path}:{line_number}: {index}")
        rows[index] = row
    return rows


class CaptionStatus:
    def __init__(self, path: Path, run_dir: Path, total: int, resume: bool):
        self.path = path
        if resume and path.is_file():
            self.payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.payload = {
                "run_dir": str(run_dir.resolve()),
                "status": "initializing",
                "phase": "initializing",
                "started_at": utc_now(),
                "completed_captions": 0,
            }
        self.payload["total_captions"] = total
        self.update()

    def update(self, **values: Any) -> None:
        if values.get("status") != "failed":
            self.payload.pop("error", None)
            self.payload.pop("traceback", None)
        self.payload.update(values)
        completed = int(self.payload.get("completed_captions", 0))
        total = int(self.payload.get("total_captions", 0))
        percent = 100.0 * completed / total if total else 100.0
        if self.payload.get("phase") == "semantic_scoring" and completed == total:
            percent = 99.9
        if self.payload.get("status") == "completed":
            percent = 100.0
        self.payload["percent"] = percent
        try:
            started = datetime.fromisoformat(str(self.payload["started_at"]))
            elapsed = max(
                0.0, (datetime.now(timezone.utc) - started).total_seconds()
            )
        except (KeyError, ValueError):
            elapsed = 0.0
        self.payload["elapsed_seconds"] = elapsed
        if 0 < completed < total and elapsed > 0.0:
            rate = completed / elapsed
            self.payload["captions_per_second"] = rate
            self.payload["eta_seconds"] = (total - completed) / rate
        elif completed >= total:
            self.payload["eta_seconds"] = 0.0
        self.payload["updated_at"] = utc_now()
        write_json(self.path, self.payload)


def ensure_protocol(path: Path, protocol: dict[str, Any]) -> bool:
    normalized = json.loads(json.dumps(protocol, ensure_ascii=True, sort_keys=True))
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise RuntimeError(f"Refusing to resume {path.parent}: protocol differs")
        return True
    write_json(path, normalized)
    return False


def caption_instruction() -> str:
    return (
        "Describe only what is visibly present in this image in one detailed, "
        "factual caption. Include visible objects, counts, colors, materials, "
        "attributes, actions, spatial relations, and rendered text when present. "
        "Do not mention this instruction and do not infer unseen information."
    )


@torch.no_grad()
def generate_caption(
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
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
    }


def retrieval_metrics(similarity: torch.Tensor) -> dict[str, float]:
    count = similarity.shape[0]
    target = torch.arange(count, device=similarity.device)
    top1 = similarity.argmax(dim=1)
    top_k = min(5, count)
    top5 = similarity.topk(top_k, dim=1).indices
    diagonal = similarity.diag().unsqueeze(1)
    ranks = (similarity > diagonal).sum(dim=1) + 1
    return {
        "top1": float((top1 == target).float().mean().cpu()),
        "top5": float((top5 == target[:, None]).any(dim=1).float().mean().cpu()),
        "mrr": float((1.0 / ranks.float()).mean().cpu()),
    }


def score_captions(
    args: argparse.Namespace,
    selected: list[tuple[int, dict[str, Any]]],
    captions: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from sacrebleu.metrics import BLEU, CHRF

    # show-o2/datasets shadows the Hugging Face datasets package imported by
    # sentence-transformers. Show-o2 inference is already released here.
    showo2_path = str(SHOWO2_ROOT)
    sys.path[:] = [value for value in sys.path if value != showo2_path]
    for module_name in list(sys.modules):
        if module_name == "datasets" or module_name.startswith("datasets."):
            del sys.modules[module_name]
    from sentence_transformers import SentenceTransformer

    prompts = [prompt_from_record(record) for _, record in selected]
    reconstructed = [captions[index]["reconstructed_caption"] for index, _ in selected]
    semantic_model = SentenceTransformer(str(args.semantic_model), device=args.device)
    prompt_embeddings = semantic_model.encode(
        prompts,
        batch_size=args.semantic_batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    caption_embeddings = semantic_model.encode(
        reconstructed,
        batch_size=args.semantic_batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    similarity = caption_embeddings @ prompt_embeddings.T
    diagonal = similarity.diag().float().cpu().tolist()
    rows = []
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (record_index, record), prompt, caption, semantic in zip(
        selected, prompts, reconstructed, diagonal
    ):
        row = {
            "record_index": record_index,
            "type": str(record.get("type", "unknown")),
            "input_prompt": prompt,
            "reconstructed_caption": caption,
            "semantic_cosine": float(semantic),
            **lexical_scores(caption, prompt),
        }
        rows.append(row)
        by_type[row["type"]].append(row)

    metric_names = (
        "semantic_cosine",
        "token_precision",
        "token_recall",
        "token_f1",
        "rouge_l_f1",
    )
    metrics = {
        name: distribution([float(row[name]) for row in rows])
        for name in metric_names
    }
    metrics["caption_to_prompt_retrieval"] = retrieval_metrics(similarity)
    metrics["prompt_to_caption_retrieval"] = retrieval_metrics(similarity.T)
    metrics["corpus_bleu"] = float(
        BLEU(effective_order=True).corpus_score(reconstructed, [prompts]).score
        / 100.0
    )
    metrics["corpus_chrf"] = float(
        CHRF().corpus_score(reconstructed, [prompts]).score / 100.0
    )
    metrics["by_type"] = {
        key: {
            "count": len(values),
            **{
                name: float(np.mean([float(row[name]) for row in values]))
                for name in metric_names
            },
        }
        for key, values in sorted(by_type.items())
    }
    del semantic_model, prompt_embeddings, caption_embeddings, similarity
    gc.collect()
    torch.cuda.empty_cache()
    return rows, metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    required = (
        args.generation_run / "protocol.json",
        args.dataset,
        args.config,
        args.vae_path,
        args.semantic_model,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing resources: {missing}")
    if args.max_caption_tokens < 1 or args.semantic_batch_size < 1:
        raise ValueError("caption token and semantic batch sizes must be positive")
    records = read_jsonl(args.dataset)
    selected = select_records(records, args)
    for record_index, _record in selected:
        if not source_image(args.generation_run, record_index).is_file():
            raise FileNotFoundError(source_image(args.generation_run, record_index))

    args.run_dir.mkdir(parents=True, exist_ok=True)
    caption_path = args.run_dir / "caption_results.jsonl"
    if args.no_resume and any(args.run_dir.iterdir()):
        raise RuntimeError("--no-resume requires an empty run directory")
    source_protocol = args.generation_run / "protocol.json"
    protocol = {
        "protocol_version": "c8-caption-cycle-v1",
        "generation_run": str(args.generation_run.resolve()),
        "generation_protocol_sha256": file_sha256(source_protocol),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "record_indices": [index for index, _ in selected],
        "config": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "model": args.model,
        "variant_name": args.variant_name,
        "drsa_checkpoint": (
            {
                "path": str(args.drsa_checkpoint.resolve()),
                "sha256": file_sha256(args.drsa_checkpoint),
            }
            if args.drsa_checkpoint else None
        ),
        "vae_path": str(args.vae_path.resolve()),
        "weight_type": args.weight_type,
        "vae_weight_type": args.vae_weight_type,
        "max_caption_tokens": args.max_caption_tokens,
        "caption_decoding": "greedy",
        "caption_prompt_has_generation_text": False,
        "image_source": "saved_png_from_generation_run",
        "image_encoding": "deterministic_vae_posterior_mean",
        "semantic_model": str(args.semantic_model.resolve()),
        "semantic_model_snapshot": args.semantic_model.name,
        "external_llm_judge": False,
    }
    resume = ensure_protocol(args.run_dir / "protocol.json", protocol)
    captions = load_captions(caption_path) if resume else {}
    selected_indices = {index for index, _ in selected}
    unexpected = set(captions) - selected_indices
    if unexpected:
        raise ValueError(f"caption results contain unexpected indices: {unexpected}")
    tracker = CaptionStatus(
        args.run_dir / "status.json",
        args.run_dir,
        len(selected),
        resume=resume,
    )
    tracker.update(
        status="running",
        phase="validated",
        completed_captions=len(captions),
    )
    if args.dry_run:
        summary = {"status": "completed", "dry_run": True, "protocol": protocol}
        write_json(args.run_dir / "summary.json", summary)
        tracker.update(status="completed", phase="dry_run_completed")
        return summary

    if len(captions) < len(selected):
        if not args.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("caption-cycle inference requires CUDA")
        from datasets.utils import image_transform
        from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
        from models.misc import get_text_tokenizer
        from omegaconf import OmegaConf
        from utils import get_hyper_params, path_to_llm_name

        config = OmegaConf.load(args.config)
        OmegaConf.resolve(config)
        device = torch.device(args.device)
        weight_type = dtype_from_name(args.weight_type)
        vae_weight_type = dtype_from_name(args.vae_weight_type)
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
        hyper = get_hyper_params(config, tokenizer, showo_token_ids)
        num_mmu_image_tokens = hyper[1]
        tracker.update(phase="loading_showo2")
        model = (
            Showo2Qwen2_5.from_pretrained(
                args.model,
                use_safetensors=False,
                local_files_only=True,
                low_cpu_mem_usage=False,
                **checkpoint_construction_kwargs(args.drsa_checkpoint),
            )
            .to(device=device, dtype=weight_type)
            .eval()
            .requires_grad_(False)
        )
        adapter = load_drsa_checkpoint(model, args.drsa_checkpoint)
        generation_adaptation = getattr(model, "_generation_adaptation", None)
        if generation_adaptation is not None:
            generation_adaptation.set_enabled(False)
            adapter["disabled_for_caption_understanding"] = True
        write_json(args.run_dir / "model_metadata.json", adapter)
        vae = WanVAE(
            vae_pth=str(args.vae_path), dtype=vae_weight_type, device=device
        )
        tracker.update(phase="caption_reconstruction")
        for record_index, record in selected:
            if record_index in captions:
                continue
            path = source_image(args.generation_run, record_index)
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            image_embeds = encode_image_for_mmu(
                image,
                model,
                vae,
                int(config.dataset.preprocessing.resolution),
                weight_type,
                device,
                image_transform,
            )
            input_embeds, attention_mask = build_mmu_prefix(
                model,
                tokenizer,
                showo_token_ids,
                image_embeds,
                caption_instruction(),
                num_mmu_image_tokens,
                add_time_embeds,
                weight_type,
                device,
                omni_attn_mask_naive,
            )
            caption = generate_caption(
                model,
                tokenizer,
                input_embeds,
                attention_mask,
                args.max_caption_tokens,
            )
            row = {
                "protocol_version": protocol["protocol_version"],
                "record_index": record_index,
                "type": str(record.get("type", "unknown")),
                "image_path": str(path.resolve()),
                "input_prompt": prompt_from_record(record),
                "input_prompt_exposed_to_captioner": False,
                "reconstructed_caption": caption,
            }
            append_jsonl(caption_path, row)
            captions[record_index] = row
            tracker.update(
                phase="caption_reconstruction",
                completed_captions=len(captions),
                current_record_index=record_index,
            )
            del image, image_embeds, input_embeds, attention_mask
            torch.cuda.empty_cache()
        del model, vae, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    tracker.update(phase="semantic_scoring", completed_captions=len(captions))
    score_rows, metrics = score_captions(args, selected, captions)
    write_jsonl(args.run_dir / "cycle_scores.jsonl", score_rows)
    summary = {
        "status": "completed",
        "completed_at": utc_now(),
        "run_dir": str(args.run_dir.resolve()),
        "variant_name": args.variant_name,
        "captions": len(captions),
        "protocol": protocol,
        "metrics": metrics,
    }
    write_json(args.run_dir / "summary.json", summary)
    tracker.update(
        status="completed",
        phase="completed",
        completed_captions=len(selected),
        summary=str((args.run_dir / "summary.json").resolve()),
    )
    return summary


def main() -> int:
    args = parse_args()
    status_path = args.run_dir / "status.json"
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
        if status_path.is_file():
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            payload.update(failure)
            write_json(status_path, payload)
        else:
            write_json(status_path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
