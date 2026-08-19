#!/usr/bin/env python3
"""Run paired baseline/guided Position generation and GenEval scoring."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
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
VARIANT_CONFIGS = {
    "baseline": {
        "guidance_steps": 0,
        "inner_steps": 5,
        "energy_step_size": 30.0,
        "layers": (7, 14, 21),
        "attention_source": "qwen",
        "max_grad_rms": 0.0,
        "max_flow_step_ratio": 0.0,
        "timestep_decay_power": 0.0,
    },
    "guided": {
        "guidance_steps": 10,
        "inner_steps": 5,
        "energy_step_size": 30.0,
        "layers": (7, 14, 21),
        "attention_source": "qwen",
        "max_grad_rms": 0.0,
        "max_flow_step_ratio": 0.0,
        "timestep_decay_power": 0.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol-version", default="quick8-v2")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=WORKSPACE_ROOT / "data/showo2_layout/position_quick8_v2.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SHOWO2_ROOT / "configs/showo2_1.5b_demo_432x432.yaml",
    )
    parser.add_argument(
        "--model", default=os.environ.get("SHOWO2_MODEL", "showlab/show-o2-1.5B")
    )
    parser.add_argument(
        "--vae-path",
        type=Path,
        default=Path(os.environ.get("SHOWO2_VAE_PATH", SHOWO2_ROOT / "Wan2.1_VAE.pth")),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--weight-type", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--vae-weight-type", choices=("bfloat16", "float32"), default="float32"
    )
    parser.add_argument("--variants", default="baseline,guided")
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument(
        "--sampling-attention-mask",
        choices=("dense", "flex"),
        default="dense",
        help=(
            "Mask used by the Euler velocity pass. Layout gradients always use "
            "a dense mask because Qwen attention export does not accept BlockMask."
        ),
    )
    parser.add_argument(
        "--verify-baseline-parity",
        action="store_true",
        help="Compare baseline against the repository's official Euler sampler.",
    )
    parser.add_argument("--inside-weight", type=float, default=1.0)
    parser.add_argument("--cross-weight", type=float, default=0.0)
    parser.add_argument("--relation-weight", type=float, default=0.0)
    parser.add_argument("--relation-margin", type=float, default=0.20)
    parser.add_argument("--relation-dominance-weight", type=float, default=0.0)
    parser.add_argument("--relation-dominance-ratio", type=float, default=0.60)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--entropy-target", type=float, default=1.0)
    parser.add_argument("--presence-weight", type=float, default=0.0)
    parser.add_argument("--presence-min", type=float, default=0.0)
    parser.add_argument("--boxdiff-inner-weight", type=float, default=0.0)
    parser.add_argument("--boxdiff-outer-weight", type=float, default=0.0)
    parser.add_argument("--boxdiff-corner-weight", type=float, default=0.0)
    parser.add_argument("--boxdiff-p", type=float, default=0.20)
    parser.add_argument("--boxdiff-corner-radius", type=int, default=1)
    parser.add_argument("--boxdiff-text-scale", type=float, default=100.0)
    parser.add_argument(
        "--boxdiff-token-mode",
        choices=("phrase_mean", "per_token"),
        default="phrase_mean",
    )
    parser.add_argument(
        "--attention-normalization",
        choices=("text_conditional", "legacy_boxdiff"),
        default="legacy_boxdiff",
    )
    parser.add_argument("--text-gate-power", type=float, default=0.0)
    parser.add_argument("--box-margin", type=float, default=0.08)
    parser.add_argument("--box-gap", type=float, default=0.10)
    parser.add_argument("--max-grad-rms", type=float, default=0.0)
    parser.add_argument("--max-flow-step-ratio", type=float, default=0.0)
    parser.add_argument("--timestep-decay-power", type=float, default=0.0)
    parser.add_argument("--backtracking-steps", type=int, default=0)
    parser.add_argument("--backtracking-factor", type=float, default=0.5)
    parser.add_argument("--min-relative-drop", type=float, default=0.0)
    parser.add_argument("--lookahead-weight", type=float, default=0.0)
    parser.add_argument("--presence-preservation-ratio", type=float, default=0.0)
    parser.add_argument("--guided-guidance-steps", type=int, default=10)
    parser.add_argument("--guided-inner-steps", type=int, default=5)
    parser.add_argument("--guided-energy-step-size", type=float, default=30.0)
    parser.add_argument("--guided-layers", default="7,14,21")
    parser.add_argument(
        "--variant-config-json",
        type=Path,
        help="Optional JSON mapping that defines additional named guidance variants.",
    )
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument(
        "--geneval-python",
        type=Path,
        default=WORKSPACE_ROOT / "venvs/geneval-score/bin/python",
    )
    parser.add_argument("--geneval-root", type=Path, default=WORKSPACE_ROOT / "geneval")
    parser.add_argument(
        "--geneval-model-config",
        type=Path,
        default=(
            WORKSPACE_ROOT / "mmdetection-2.28.2/mmdet/.mim/configs/mask2former/"
            "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
        ),
    )
    parser.add_argument(
        "--geneval-model-path",
        type=Path,
        default=WORKSPACE_ROOT / "models/geneval",
    )
    parser.add_argument(
        "--geneval-clip-path",
        type=Path,
        default=WORKSPACE_ROOT / "models/geneval/clip/ViT-L-14.state_dict.pt",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )


def ensure_protocol(path: Path, protocol: dict[str, Any]) -> bool:
    """Persist a new protocol or verify that an existing run is identical."""
    normalized = json.loads(json.dumps(protocol, ensure_ascii=True, sort_keys=True))
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise RuntimeError(
                f"Refusing to resume {path.parent}: protocol.json differs from "
                "the requested experiment"
            )
        return True
    write_json(path, normalized)
    return False


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def generation_prompt(record: dict) -> str:
    prompt = str(record.get("generation_prompt", record["prompt"])).strip()
    if not prompt:
        raise ValueError(f"Empty generation prompt in record {record}")
    return prompt


def dtype_from_name(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def parse_variants(value: str) -> tuple[str, ...]:
    variants = tuple(item.strip() for item in value.split(",") if item.strip())
    if not variants or len(set(variants)) != len(variants):
        raise ValueError(f"Invalid variant list: {value!r}")
    unknown = set(variants) - set(VARIANT_CONFIGS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    return variants


def load_variant_configs(path: Path | None) -> None:
    if path is None:
        return
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("variant config JSON must be a non-empty object")
    required = {
        "guidance_steps",
        "inner_steps",
        "energy_step_size",
        "layers",
        "attention_source",
        "max_grad_rms",
    }
    for name, raw_config in payload.items():
        if not isinstance(name, str) or not name or not isinstance(raw_config, dict):
            raise ValueError("variant names must map to JSON objects")
        missing = required - set(raw_config)
        if missing:
            raise ValueError(f"Variant {name!r} is missing fields: {sorted(missing)}")
        layers = tuple(int(index) for index in raw_config["layers"])
        config = {
            "guidance_steps": int(raw_config["guidance_steps"]),
            "inner_steps": int(raw_config["inner_steps"]),
            "energy_step_size": float(raw_config["energy_step_size"]),
            "layers": layers,
            "attention_source": str(raw_config["attention_source"]),
            "max_grad_rms": float(raw_config["max_grad_rms"]),
            "max_flow_step_ratio": float(
                raw_config.get("max_flow_step_ratio", 0.0)
            ),
            "timestep_decay_power": float(
                raw_config.get("timestep_decay_power", 0.0)
            ),
            "backtracking_steps": int(raw_config.get("backtracking_steps", 0)),
            "backtracking_factor": float(
                raw_config.get("backtracking_factor", 0.5)
            ),
            "min_relative_drop": float(
                raw_config.get("min_relative_drop", 0.0)
            ),
            "lookahead_weight": float(raw_config.get("lookahead_weight", 0.0)),
            "presence_preservation_ratio": float(
                raw_config.get("presence_preservation_ratio", 0.0)
            ),
            "drsa_checkpoint": (
                str(Path(raw_config["drsa_checkpoint"]).expanduser().resolve())
                if raw_config.get("drsa_checkpoint") else None
            ),
        }
        if config["guidance_steps"] < 0:
            raise ValueError(f"Variant {name!r} guidance_steps must be non-negative")
        if config["inner_steps"] < 1 or config["energy_step_size"] <= 0.0:
            raise ValueError(f"Variant {name!r} inner steps and step size must be positive")
        if not layers:
            raise ValueError(f"Variant {name!r} must select at least one layer")
        if config["attention_source"] not in {"qwen", "diffusion"}:
            raise ValueError(f"Variant {name!r} has invalid attention_source")
        if config["max_grad_rms"] < 0.0:
            raise ValueError(f"Variant {name!r} max_grad_rms must be non-negative")
        if config["max_flow_step_ratio"] < 0.0:
            raise ValueError(
                f"Variant {name!r} max_flow_step_ratio must be non-negative"
            )
        if config["timestep_decay_power"] < 0.0:
            raise ValueError(
                f"Variant {name!r} timestep_decay_power must be non-negative"
            )
        if config["backtracking_steps"] < 0:
            raise ValueError(
                f"Variant {name!r} backtracking_steps must be non-negative"
            )
        if not 0.0 < config["backtracking_factor"] < 1.0:
            raise ValueError(
                f"Variant {name!r} backtracking_factor must be in (0, 1)"
            )
        if config["min_relative_drop"] < 0.0:
            raise ValueError(
                f"Variant {name!r} min_relative_drop must be non-negative"
            )
        if config["lookahead_weight"] < 0.0:
            raise ValueError(
                f"Variant {name!r} lookahead_weight must be non-negative"
            )
        if not 0.0 <= config["presence_preservation_ratio"] <= 1.0:
            raise ValueError(
                f"Variant {name!r} presence_preservation_ratio must be in [0, 1]"
            )
        if (
            config["lookahead_weight"] > 0.0
            and config["backtracking_steps"] == 0
        ):
            raise ValueError(
                f"Variant {name!r} lookahead requires backtracking"
            )
        VARIANT_CONFIGS[name] = config


def sample_seed(base_seed: int, source_index: int, sample_index: int) -> int:
    return base_seed + source_index * 10 + sample_index


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_construction_kwargs(
    checkpoint_paths: list[str | None],
) -> dict[str, Any]:
    from models.generation_adaptation import checkpoint_model_kwargs

    configs = []
    for checkpoint_path in checkpoint_paths:
        if checkpoint_path is None:
            continue
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        configs.append(checkpoint_model_kwargs(payload))
        del payload
    if not configs:
        return {"drsa_enabled": False, "generation_adapter_enabled": False}
    unique = {tuple(sorted(item.items())) for item in configs}
    if len(unique) != 1:
        raise ValueError(
            "one GenEval process cannot mix legacy DRSA, G-adapter, and LoRA checkpoints"
        )
    return configs[0]


def load_drsa_checkpoint(model, checkpoint_path: str | None) -> dict[str, Any]:
    """Load legacy DRSA or a new generation-only adaptation checkpoint."""
    if checkpoint_path is None:
        adaptation = getattr(model, "_generation_adaptation", None)
        if adaptation is not None:
            adaptation.set_enabled(False)
        return {"enabled": False, "checkpoint": None}
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "adaptation_config" in payload:
        from models.generation_adaptation import configure_from_checkpoint

        route = str(payload["adaptation_config"]["route"])
        adaptation = getattr(model, "_generation_adaptation", None)
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
        missing_keys: list[str] = []
        unexpected_keys: list[str] = []
    else:
        if model.drsa_adapter is None:
            raise RuntimeError("DRSA checkpoint requested but the model has no adapter")
        if "drsa_adapter" not in payload or "step" not in payload:
            raise KeyError(f"Invalid checkpoint: {path}")
        state = payload["drsa_adapter"]
        if any(key.startswith("generation_spatial.") for key in state):
            model.drsa_adapter.enable_generation_spatial()
        incompatible = model.drsa_adapter.load_state_dict(state, strict=True)
        parameter_norm = math.sqrt(sum(
            float(value.detach().float().square().sum())
            for value in model.drsa_adapter.state_dict().values()
        ))
        kind = "legacy_drsa"
        requires_relation_code = True
        route_config = None
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
    metadata = {
        "enabled": True,
        "kind": kind,
        "route": route_config,
        "requires_relation_code": requires_relation_code,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": file_sha256(path),
        "step": int(payload["step"]),
        "parameter_l2": parameter_norm,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }
    del payload
    if not math.isfinite(parameter_norm) or parameter_norm <= 0.0:
        raise RuntimeError(f"Invalid DRSA parameter norm: {parameter_norm}")
    return metadata


def encode_drsa_relation(model, tokenizer, relation: str, batch_size: int, cfg: bool):
    encoded = tokenizer(
        [relation], add_special_tokens=False, truncation=True, max_length=32,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    features = model.showo.model.embed_tokens(token_ids)
    code = model.drsa_adapter.encode_relation(features, attention_mask)
    code = code.repeat(batch_size, 1)
    if cfg:
        code = torch.cat([code, code], dim=0)
    return code

def validate_manifest(
    records: list[dict], samples_per_prompt: int, base_seed: int
) -> dict:
    if not records:
        raise ValueError("The Position manifest must not be empty")
    local_indices = [int(item["local_index"]) for item in records]
    if local_indices != list(range(len(records))):
        raise ValueError(
            f"local_index must be contiguous and ordered, got {local_indices}"
        )
    counts = Counter(item["relation"] for item in records)
    unknown_relations = set(counts) - {"left of", "right of", "above", "below"}
    if unknown_relations:
        raise ValueError(f"Unsupported Position relations: {sorted(unknown_relations)}")
    if samples_per_prompt != 4:
        raise ValueError(
            "The paired Position protocol requires exactly four samples per prompt"
        )

    if str(SHOWO2_ROOT) not in sys.path:
        sys.path.insert(0, str(SHOWO2_ROOT))
    from guidance import parse_position_layout

    seeds = []
    for record in records:
        layout = parse_position_layout(generation_prompt(record))
        expected = str(record["relation"]).replace(" of", "")
        if layout.relation != expected:
            raise ValueError(
                f"Layout parser mismatch at source {record['source_index']}: "
                f"{layout.relation} != {expected}"
            )
        metadata = record["metadata"]
        if (
            metadata.get("tag") != "position"
            or not metadata.get("prompt")
        ):
            raise ValueError(
                f"Invalid GenEval metadata at source {record['source_index']}"
            )
        for index in range(samples_per_prompt):
            seeds.append(sample_seed(base_seed, int(record["source_index"]), index))
    if len(seeds) != len(set(seeds)):
        raise ValueError(
            "Different prompt/sample pairs must not reuse initial-noise seeds"
        )
    return {"relations": counts, "seeds": seeds}


class StatusTracker:
    def __init__(
        self, path: Path, run_dir: Path, total_images: int, resume: bool = False
    ):
        self.path = path
        if resume and path.is_file():
            self.payload = json.loads(path.read_text(encoding="utf-8"))
            return
        self.payload = {
            "run_dir": str(run_dir.resolve()),
            "status": "initializing",
            "phase": "initializing",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "completed_images": 0,
            "total_images": total_images,
        }
        self.update()

    def update(self, **values: Any) -> None:
        if values.get("status") != "failed":
            self.payload.pop("error", None)
            self.payload.pop("traceback", None)
        self.payload.update(values)
        self.payload["updated_at"] = utc_now()
        write_json(self.path, self.payload)


def count_images(run_dir: Path, variants: tuple[str, ...]) -> int:
    return sum(
        1
        for variant in variants
        for path in (run_dir / variant / "geneval").glob("*/samples/*.png")
        if path.is_file()
    )


def latent_hash(latent: torch.Tensor) -> str:
    array = latent.detach().float().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def merge_pair_hash(
    hashes: dict[str, dict[str, str]],
    key: str,
    variant: str,
    digest: str,
) -> None:
    values = hashes.setdefault(key, {})
    previous = values.setdefault(variant, digest)
    if previous != digest:
        raise RuntimeError(
            f"Conflicting initial latent hash for {key}/{variant}: "
            f"{previous} != {digest}"
        )
    other_digests = {value for name, value in values.items() if name != variant}
    if other_digests and other_digests != {digest}:
        raise RuntimeError(
            f"Initial latent mismatch across variants for {key}: {values}"
        )


def load_recorded_pair_hashes(
    run_dir: Path,
    variants: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    hashes: dict[str, dict[str, str]] = {}
    persisted = run_dir / "paired_initial_latents.json"
    if persisted.is_file():
        payload = json.loads(persisted.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Invalid pairing payload in {persisted}")
        for key, values in payload.items():
            if not isinstance(values, dict):
                raise TypeError(f"Invalid pairing record for {key!r}")
            for variant, digest in values.items():
                if variant in variants:
                    merge_pair_hash(hashes, str(key), str(variant), str(digest))

    for variant in variants:
        for path in sorted((run_dir / variant / "raw").glob("*/run-*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            source_index = int(payload["source_index"])
            sample_indices = payload["sample_indices"]
            digests = payload["initial_latent_sha256"]
            if len(sample_indices) != len(digests):
                raise ValueError(f"Mismatched samples and hashes in {path}")
            for sample_index, digest in zip(sample_indices, digests):
                merge_pair_hash(
                    hashes,
                    f"{source_index}:{int(sample_index)}",
                    variant,
                    str(digest),
                )
    return hashes


def pending_sample_indices(
    prompt_dir: Path,
    pair_hashes: dict[str, dict[str, str]],
    source_index: int,
    variant: str,
    samples_per_prompt: int,
) -> list[int]:
    pending = []
    for sample_index in range(samples_per_prompt):
        image = prompt_dir / "samples" / f"{sample_index:05d}.png"
        key = f"{source_index}:{sample_index}"
        paired = variant in pair_hashes.get(key, {})
        if not image.is_file() or not paired:
            pending.append(sample_index)
    return pending


def seeded_latents(
    seeds: list[int],
    latent_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[str]]:
    tensors = []
    hashes = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        latent = torch.randn(
            latent_shape, generator=generator, device=device, dtype=dtype
        )
        tensors.append(latent)
        hashes.append(latent_hash(latent))
    return torch.cat(tensors, dim=0), hashes


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_metadata(record: dict, variant_dir: Path) -> Path:
    prompt_dir = variant_dir / "geneval" / f"{int(record['local_index']):05d}"
    (prompt_dir / "samples").mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        prompt_dir / "metadata.jsonl",
        json.dumps(record["metadata"], ensure_ascii=True, sort_keys=True) + "\n",
    )
    return prompt_dir


def run_generation(
    args: argparse.Namespace,
    records: list[dict],
    variants: tuple[str, ...],
    tracker: StatusTracker,
) -> None:
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.vae_path.is_file():
        raise FileNotFoundError(args.vae_path)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("quick8-v2 generation requires CUDA")

    if str(SHOWO2_ROOT) not in sys.path:
        sys.path.insert(0, str(SHOWO2_ROOT))
    from omegaconf import OmegaConf
    from torch.nn.attention.flex_attention import create_block_mask

    from guidance import EnergyConfig, GuidedEulerSampler, GuiderConfig
    from guidance import LatentLayoutGuider, LayoutEnergy, map_layout_tokens
    from guidance import layout_to_dict, parse_position_layout
    from models import (
        Showo2Qwen2_5,
        WanVAE,
        omni_attn_mask,
        omni_attn_mask_naive,
    )
    from models.misc import get_text_tokenizer, prepare_gen_input
    from transport import Sampler, create_transport
    from utils import denorm, get_hyper_params, path_to_llm_name

    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)
    weight_type = dtype_from_name(args.weight_type)
    vae_weight_type = dtype_from_name(args.vae_weight_type)
    device = torch.device(args.device)
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
        _default_cfg_scale,
    ) = get_hyper_params(config, tokenizer, showo_token_ids)
    time_shifting_factor = (
        float(config.transport.time_shifting_factor)
        if config.transport.time_shifting_factor is not None
        else None
    )

    tracker.update(status="running", phase="loading_model")
    checkpoint_kwargs = checkpoint_construction_kwargs(
        [VARIANT_CONFIGS[variant].get("drsa_checkpoint") for variant in variants]
    )
    print(json.dumps({"status": "loading_model", "model": args.model}), flush=True)
    model = (
        Showo2Qwen2_5.from_pretrained(
            args.model,
            use_safetensors=False,
            local_files_only=True,
            low_cpu_mem_usage=not any(checkpoint_kwargs.values()),
            **checkpoint_kwargs,
        )
        .to(device=device, dtype=weight_type)
        .eval()
        .requires_grad_(False)
    )
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
    latent_shape = (
        1,
        image_latent_dim,
        latent_height * patch_size,
        latent_width * patch_size,
    )

    pair_hashes = load_recorded_pair_hashes(args.run_dir, variants)
    for variant in variants:
        variant_config = VARIANT_CONFIGS[variant]
        variant_dir = args.run_dir / variant
        adapter_metadata = load_drsa_checkpoint(
            model, variant_config.get("drsa_checkpoint")
        )
        write_json(variant_dir / "drsa_checkpoint.json", adapter_metadata)
        for record in records:
            prompt_dir = prepare_metadata(record, variant_dir)
            sample_indices = pending_sample_indices(
                prompt_dir,
                pair_hashes,
                int(record["source_index"]),
                variant,
                args.samples_per_prompt,
            )
            if not sample_indices:
                print(
                    json.dumps(
                        {
                            "status": "skipped",
                            "variant": variant,
                            "source_index": record["source_index"],
                            "reason": "all_samples_exist",
                        }
                    ),
                    flush=True,
                )
                continue

            prompt = generation_prompt(record)
            layout = map_layout_tokens(
                parse_position_layout(
                    prompt,
                    margin=args.box_margin,
                    gap=args.box_gap,
                ),
                tokenizer,
            )
            for chunk_start in range(0, len(sample_indices), args.sample_batch_size):
                chunk = sample_indices[
                    chunk_start : chunk_start + args.sample_batch_size
                ]
                seeds = [
                    sample_seed(args.base_seed, int(record["source_index"]), index)
                    for index in chunk
                ]
                batch_size = len(chunk)
                tracker.update(
                    status="running",
                    phase="generation",
                    variant=variant,
                    local_index=int(record["local_index"]),
                    source_index=int(record["source_index"]),
                    relation=record["relation"],
                    sample_indices=chunk,
                    euler_step=0,
                    euler_steps=args.num_inference_steps - 1,
                    completed_images=count_images(args.run_dir, variants),
                )
                print(
                    json.dumps(
                        {
                            "status": "generating",
                            "variant": variant,
                            "source_index": record["source_index"],
                            "samples": chunk,
                            "seeds": seeds,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

                text_cond, text_null, positions_cond, positions_null = (
                    prepare_gen_input(
                        [prompt] * batch_size,
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
                )
                initial_latents, hashes = seeded_latents(
                    seeds, latent_shape, device, weight_type
                )
                for sample_index, digest in zip(chunk, hashes):
                    key = f"{int(record['source_index'])}:{sample_index}"
                    merge_pair_hash(pair_hashes, key, variant, digest)

                if args.guidance_scale > 0.0:
                    initial_latents = torch.cat(
                        [initial_latents, initial_latents], dim=0
                    )
                    text_tokens = torch.cat([text_cond, text_null], dim=0)
                    modality_positions = torch.cat(
                        [positions_cond, positions_null], dim=0
                    )
                else:
                    text_tokens = text_cond
                    modality_positions = positions_cond
                dense_attention_mask = omni_attn_mask_naive(
                    text_tokens.size(0), max_seq_len, modality_positions, device
                ).to(weight_type)
                model_kwargs = {
                    "text_tokens": text_tokens,
                    "attention_mask": dense_attention_mask,
                    "modality_positions": modality_positions,
                    "output_hidden_states": True,
                    "max_seq_len": max_seq_len,
                    "guidance_scale": args.guidance_scale,
                }
                relation_code = None
                if adapter_metadata.get("requires_relation_code", False):
                    relation_code = encode_drsa_relation(
                        model, tokenizer, str(record["relation"]), batch_size,
                        cfg=args.guidance_scale > 0.0,
                    )
                    if relation_code is None or not torch.isfinite(relation_code).all():
                        raise RuntimeError("DRSA relation code is missing or non-finite")
                    if int(record["local_index"]) == 0 and chunk_start == 0:
                        print(json.dumps({
                            "status": "drsa_active",
                            "variant": variant,
                            "relation_code_l2": float(relation_code.float().norm().cpu()),
                            "checkpoint_step": adapter_metadata["step"],
                        }, sort_keys=True), flush=True)
                    model_kwargs["drsa_relation_codes"] = relation_code
                velocity_model_kwargs = model_kwargs
                if args.sampling_attention_mask == "flex":
                    velocity_model_kwargs = dict(model_kwargs)
                    mask_fn = omni_attn_mask(modality_positions)
                    velocity_model_kwargs["attention_mask"] = create_block_mask(
                        mask_fn,
                        B=text_tokens.size(0),
                        H=None,
                        Q_LEN=max_seq_len,
                        KV_LEN=max_seq_len,
                        device=device,
                    )
                energy = LayoutEnergy(
                    EnergyConfig(
                        layers=variant_config["layers"],
                        inside_weight=args.inside_weight,
                        cross_weight=args.cross_weight,
                        relation_weight=args.relation_weight,
                        relation_margin=args.relation_margin,
                        relation_dominance_weight=args.relation_dominance_weight,
                        relation_dominance_ratio=args.relation_dominance_ratio,
                        entropy_weight=args.entropy_weight,
                        entropy_target=args.entropy_target,
                        presence_weight=args.presence_weight,
                        presence_min=args.presence_min,
                        boxdiff_inner_weight=args.boxdiff_inner_weight,
                        boxdiff_outer_weight=args.boxdiff_outer_weight,
                        boxdiff_corner_weight=args.boxdiff_corner_weight,
                        boxdiff_p=args.boxdiff_p,
                        boxdiff_corner_radius=args.boxdiff_corner_radius,
                        boxdiff_text_scale=args.boxdiff_text_scale,
                        boxdiff_token_mode=args.boxdiff_token_mode,
                        attention_normalization=args.attention_normalization,
                        text_gate_power=args.text_gate_power,
                    )
                )
                guider = LatentLayoutGuider(
                    model=model,
                    energy=energy,
                    layouts=(layout,) * batch_size,
                    modality_positions=positions_cond,
                    spatial_shape=(latent_height, latent_width),
                    add_time_embeds=add_time_embeds,
                    config=GuiderConfig(
                        inner_steps=variant_config["inner_steps"],
                        step_size=variant_config["energy_step_size"],
                        max_grad_rms=variant_config.get(
                            "max_grad_rms", args.max_grad_rms
                        ),
                        max_flow_step_ratio=variant_config.get(
                            "max_flow_step_ratio", args.max_flow_step_ratio
                        ),
                        timestep_decay_power=variant_config.get(
                            "timestep_decay_power", args.timestep_decay_power
                        ),
                        attention_source=variant_config.get(
                            "attention_source", "qwen"
                        ),
                        backtracking_steps=variant_config.get(
                            "backtracking_steps", args.backtracking_steps
                        ),
                        backtracking_factor=variant_config.get(
                            "backtracking_factor", args.backtracking_factor
                        ),
                        min_relative_drop=variant_config.get(
                            "min_relative_drop", args.min_relative_drop
                        ),
                        lookahead_weight=variant_config.get(
                            "lookahead_weight", args.lookahead_weight
                        ),
                        presence_preservation_ratio=variant_config.get(
                            "presence_preservation_ratio",
                            args.presence_preservation_ratio,
                        ),
                    ),
                )
                sampler = GuidedEulerSampler(
                    transport=transport,
                    num_steps=args.num_inference_steps,
                    guidance_steps=variant_config["guidance_steps"],
                    time_shifting_factor=time_shifting_factor,
                    reverse=bool(config.transport.reverse),
                )

                def report(step_record: dict[str, Any]) -> None:
                    step = int(step_record["step"]) + 1
                    tracker.update(euler_step=step)
                    if (
                        step == 1
                        or step % 5 == 0
                        or step == args.num_inference_steps - 1
                    ):
                        energy_value = None
                        if step_record["guidance"]:
                            energy_value = step_record["guidance"][-1]["energy"]
                        print(
                            json.dumps(
                                {
                                    "status": "sampling",
                                    "variant": variant,
                                    "source_index": record["source_index"],
                                    "step": step,
                                    "total": args.num_inference_steps - 1,
                                    "guided": bool(step_record["guidance"]),
                                    "energy": energy_value,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )

                final_latents, diagnostics = sampler.sample(
                    initial_latents=initial_latents,
                    model=model,
                    model_kwargs=model_kwargs,
                    guider=guider,
                    velocity_model_kwargs=velocity_model_kwargs,
                    progress=report,
                )
                parity = None
                if args.verify_baseline_parity and variant == "baseline":
                    official_sample = Sampler(transport).sample_ode(
                        sampling_method=config.transport.sampling_method,
                        num_steps=args.num_inference_steps,
                        atol=config.transport.atol,
                        rtol=config.transport.rtol,
                        reverse=bool(config.transport.reverse),
                        time_shifting_factor=time_shifting_factor,
                    )
                    reference = official_sample(
                        initial_latents,
                        model.t2i_generate,
                        **velocity_model_kwargs,
                    )[-1]
                    difference = (final_latents - reference).float()
                    parity = {
                        "max_abs": float(difference.abs().max().cpu()),
                        "rms": float(difference.square().mean().sqrt().cpu()),
                        "reference_rms": float(
                            reference.float().square().mean().sqrt().cpu()
                        ),
                    }
                    if parity["max_abs"] > 1e-6:
                        raise RuntimeError(
                            "Custom baseline diverged from official Euler sampler: "
                            f"{parity}"
                        )
                    del reference, difference
                if args.guidance_scale > 0.0:
                    final_latents = torch.chunk(final_latents, 2, dim=0)[0]
                with torch.no_grad():
                    decoded = vae.batch_decode(final_latents.unsqueeze(2)).squeeze(2)
                arrays = denorm(decoded)
                for array, sample_index in zip(arrays, chunk):
                    Image.fromarray(array).save(
                        prompt_dir / "samples" / f"{sample_index:05d}.png"
                    )

                raw_dir = variant_dir / "raw" / f"{int(record['local_index']):05d}"
                batch_name = "-".join(f"{item:05d}" for item in chunk)
                write_json(raw_dir / f"diagnostics-{batch_name}.json", diagnostics)
                write_json(
                    raw_dir / f"run-{batch_name}.json",
                    {
                        "status": "completed",
                        "variant": variant,
                        "local_index": int(record["local_index"]),
                        "source_index": int(record["source_index"]),
                        "relation": record["relation"],
                        "prompt": prompt,
                        "scoring_prompt": record["metadata"].get("prompt"),
                        "sample_indices": chunk,
                        "seeds": seeds,
                        "initial_latent_sha256": hashes,
                        "guidance_scale": args.guidance_scale,
                        "num_inference_steps": args.num_inference_steps,
                        "sampling_attention_mask": args.sampling_attention_mask,
                        "baseline_parity": parity,
                        "drsa": adapter_metadata,
                        "relation_code_l2": float(relation_code.float().norm().cpu()) if relation_code is not None else None,
                        "guidance": variant_config,
                        "energy": {
                            "inside_weight": args.inside_weight,
                            "cross_weight": args.cross_weight,
                            "relation_weight": args.relation_weight,
                            "relation_margin": args.relation_margin,
                            "relation_dominance_weight": args.relation_dominance_weight,
                            "relation_dominance_ratio": args.relation_dominance_ratio,
                            "entropy_weight": args.entropy_weight,
                            "entropy_target": args.entropy_target,
                            "presence_weight": args.presence_weight,
                            "presence_min": args.presence_min,
                            "boxdiff_inner_weight": args.boxdiff_inner_weight,
                            "boxdiff_outer_weight": args.boxdiff_outer_weight,
                            "boxdiff_corner_weight": args.boxdiff_corner_weight,
                            "boxdiff_p": args.boxdiff_p,
                            "boxdiff_corner_radius": args.boxdiff_corner_radius,
                            "boxdiff_text_scale": args.boxdiff_text_scale,
                            "boxdiff_token_mode": args.boxdiff_token_mode,
                            "attention_normalization": args.attention_normalization,
                            "text_gate_power": args.text_gate_power,
                        },
                        "layout": layout_to_dict(layout),
                    },
                )
                write_json(args.run_dir / "paired_initial_latents.json", pair_hashes)
                tracker.update(completed_images=count_images(args.run_dir, variants))
                del decoded, final_latents, initial_latents, model_kwargs
                del dense_attention_mask, text_tokens, modality_positions
                del guider, sampler, velocity_model_kwargs
                torch.cuda.empty_cache()

    write_json(args.run_dir / "paired_initial_latents.json", pair_hashes)
    tracker.update(
        status="running",
        phase="releasing_generation_models",
        completed_images=count_images(args.run_dir, variants),
    )
    del model, vae, transport, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def score_variant(
    args: argparse.Namespace,
    variant: str,
    tracker: StatusTracker,
    variants: tuple[str, ...],
) -> None:
    evaluate_script = args.geneval_root / "evaluation/evaluate_images.py"
    summary_script = args.geneval_root / "evaluation/summary_scores.py"
    required = (
        args.geneval_python,
        evaluate_script,
        summary_script,
        args.geneval_model_config,
        args.geneval_clip_path,
        args.geneval_model_path / "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing GenEval scoring dependencies: {missing}")

    variant_dir = args.run_dir / variant
    results_dir = variant_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "results.jsonl"
    score_log = results_dir / "score.log"
    expected_images = count_images(args.run_dir, (variant,))
    if results_file.is_file() and expected_images > 0:
        cached_lines = sum(
            1 for line in results_file.read_text(encoding="utf-8").splitlines() if line
        )
        if cached_lines == expected_images:
            tracker.update(
                status="running",
                phase="scoring_cached",
                variant=variant,
                completed_images=count_images(args.run_dir, variants),
                euler_step=None,
                euler_steps=None,
            )
            print(json.dumps({
                "status": "scoring_cached",
                "variant": variant,
                "images": expected_images,
            }, sort_keys=True), flush=True)
    tracker.update(
        status="running",
        phase="scoring",
        variant=variant,
        completed_images=count_images(args.run_dir, variants),
        euler_step=None,
        euler_steps=None,
    )
    command = [
        str(args.geneval_python),
        str(evaluate_script),
        str(variant_dir / "geneval"),
        "--outfile",
        str(results_file),
        "--model-config",
        str(args.geneval_model_config),
        "--model-path",
        str(args.geneval_model_path),
        "--clip-path",
        str(args.geneval_clip_path),
        "--options",
        "num_workers=0",
    ]
    with score_log.open("w", encoding="utf-8") as log_handle:
        subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, check=True)
    summary = subprocess.run(
        [str(args.geneval_python), str(summary_script), str(results_file)],
        capture_output=True,
        text=True,
        check=True,
    )
    atomic_write_text(results_dir / "summary.txt", summary.stdout + summary.stderr)
    line_count = sum(
        1 for line in results_file.read_text(encoding="utf-8").splitlines() if line
    )
    expected = count_images(args.run_dir, (variant,))
    if expected < 1 or line_count != expected:
        raise RuntimeError(
            f"Expected {expected} scored images for {variant}, got {line_count}"
        )


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.manifest = args.manifest.resolve()
    load_variant_configs(args.variant_config_json)
    try:
        guided_layers = tuple(
            int(item.strip()) for item in args.guided_layers.split(",") if item.strip()
        )
    except ValueError as exc:
        raise ValueError("guided_layers must be comma-separated integers") from exc
    if not guided_layers:
        raise ValueError("guided_layers must not be empty")
    VARIANT_CONFIGS["guided"] = {
        "guidance_steps": args.guided_guidance_steps,
        "inner_steps": args.guided_inner_steps,
        "energy_step_size": args.guided_energy_step_size,
        "layers": guided_layers,
        "attention_source": "qwen",
        "max_grad_rms": args.max_grad_rms,
        "max_flow_step_ratio": args.max_flow_step_ratio,
        "timestep_decay_power": args.timestep_decay_power,
        "backtracking_steps": args.backtracking_steps,
        "backtracking_factor": args.backtracking_factor,
        "min_relative_drop": args.min_relative_drop,
        "lookahead_weight": args.lookahead_weight,
        "presence_preservation_ratio": args.presence_preservation_ratio,
    }
    if not 0 <= args.guided_guidance_steps < args.num_inference_steps:
        raise ValueError("guided_guidance_steps must be below num_inference_steps")
    if args.guided_inner_steps < 1 or args.guided_energy_step_size <= 0.0:
        raise ValueError("guided inner steps and energy step size must be positive")
    variants = parse_variants(args.variants)
    if args.sampling_attention_mask != "flex" and any(
        VARIANT_CONFIGS[variant].get("backtracking_steps", 0) > 0
        for variant in variants
    ):
        raise ValueError(
            "Backtracking variants require --sampling-attention-mask flex"
        )
    records = read_jsonl(args.manifest)
    if args.limit_prompts is not None:
        if args.limit_prompts < 1:
            raise ValueError("limit_prompts must be positive")
        records = records[:args.limit_prompts]
    validation = validate_manifest(records, args.samples_per_prompt, args.base_seed)
    if args.sample_batch_size < 1 or args.sample_batch_size > args.samples_per_prompt:
        raise ValueError("sample_batch_size must be between 1 and samples_per_prompt")


    for variant in variants:
        checkpoint = VARIANT_CONFIGS[variant].get("drsa_checkpoint")
        if checkpoint is not None:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            VARIANT_CONFIGS[variant]["drsa_checkpoint_sha256"] = file_sha256(
                checkpoint_path
            )
    protocol = {
        "protocol_version": args.protocol_version,
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "energy": {
            "inside_weight": args.inside_weight,
            "cross_weight": args.cross_weight,
            "relation_weight": args.relation_weight,
            "relation_margin": args.relation_margin,
            "relation_dominance_weight": args.relation_dominance_weight,
            "relation_dominance_ratio": args.relation_dominance_ratio,
            "entropy_weight": args.entropy_weight,
            "entropy_target": args.entropy_target,
            "presence_weight": args.presence_weight,
            "presence_min": args.presence_min,
            "boxdiff_inner_weight": args.boxdiff_inner_weight,
            "boxdiff_outer_weight": args.boxdiff_outer_weight,
            "boxdiff_corner_weight": args.boxdiff_corner_weight,
            "boxdiff_p": args.boxdiff_p,
            "boxdiff_corner_radius": args.boxdiff_corner_radius,
            "boxdiff_text_scale": args.boxdiff_text_scale,
            "boxdiff_token_mode": args.boxdiff_token_mode,
            "attention_normalization": args.attention_normalization,
            "text_gate_power": args.text_gate_power,
        },
        "variants": {variant: VARIANT_CONFIGS[variant] for variant in variants},
        "relations": validation["relations"],
        "prompts": len(records),
        "samples_per_prompt": args.samples_per_prompt,
        "total_images": len(records) * args.samples_per_prompt * len(variants),
        "seed_formula": "base_seed + source_index * 10 + sample_index",
        "base_seed": args.base_seed,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "sampling_attention_mask": args.sampling_attention_mask,
        "verify_baseline_parity": args.verify_baseline_parity,
        "sample_batch_size": args.sample_batch_size,
        "limit_prompts": args.limit_prompts,
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "mode": "dry_run",
                    "protocol": protocol,
                    "seeds": validation["seeds"],
                },
                default=list,
                sort_keys=True,
            )
        )
        return

    args.run_dir.mkdir(parents=True, exist_ok=True)
    resume = ensure_protocol(args.run_dir / "protocol.json", protocol)
    tracker = StatusTracker(
        args.run_dir / "status.json",
        args.run_dir,
        protocol["total_images"],
        resume=resume,
    )
    try:
        run_generation(args, records, variants, tracker)
        if not args.skip_score:
            for variant in variants:
                score_variant(args, variant, tracker, variants)
        tracker.update(
            status="completed",
            phase="completed",
            completed_images=count_images(args.run_dir, variants),
            variant=None,
        )
        print(json.dumps({"status": "completed", "run_dir": str(args.run_dir)}))
    except BaseException as exc:
        tracker.update(
            status="failed",
            phase="failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            completed_images=count_images(args.run_dir, variants),
        )
        raise


if __name__ == "__main__":
    main()
