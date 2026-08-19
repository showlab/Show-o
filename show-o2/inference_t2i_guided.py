#!/usr/bin/env python3
"""Independent training-free layout-guided T2I inference for Show-o2."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent


def parse_layers(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "layers must be comma-separated integers"
        ) from exc
    if not layers:
        raise argparse.ArgumentTypeError("at least one layer is required")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/showo2_1.5b_demo_432x432.yaml",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("SHOWO2_MODEL", "showlab/show-o2-1.5B"),
    )
    parser.add_argument(
        "--vae-path",
        type=Path,
        default=Path(os.environ.get("SHOWO2_VAE_PATH", str(ROOT / "Wan2.1_VAE.pth"))),
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--layout-json",
        help="A JSON file or inline JSON object with phrase/box constraints",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--weight-type",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--vae-weight-type",
        choices=("bfloat16", "float32"),
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--guidance-steps", type=int, default=10)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--energy-step-size", type=float, default=30.0)
    parser.add_argument("--max-grad-rms", type=float, default=0.0)
    parser.add_argument("--max-flow-step-ratio", type=float, default=0.0)
    parser.add_argument("--timestep-decay-power", type=float, default=0.0)
    parser.add_argument("--backtracking-steps", type=int, default=0)
    parser.add_argument("--backtracking-factor", type=float, default=0.5)
    parser.add_argument("--min-relative-drop", type=float, default=0.0)
    parser.add_argument("--lookahead-weight", type=float, default=0.0)
    parser.add_argument("--presence-preservation-ratio", type=float, default=0.0)
    parser.add_argument(
        "--sampling-attention-mask",
        choices=("dense", "flex"),
        default="dense",
    )
    parser.add_argument("--layers", type=parse_layers, default=(7, 14, 21))
    parser.add_argument(
        "--attention-source",
        choices=("qwen", "diffusion"),
        default="qwen",
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
        default="per_token",
    )
    parser.add_argument(
        "--attention-normalization",
        choices=("text_conditional", "legacy_boxdiff"),
        default="text_conditional",
    )
    parser.add_argument("--text-gate-power", type=float, default=0.0)
    parser.add_argument("--box-margin", type=float, default=0.08)
    parser.add_argument("--box-gap", type=float, default=0.10)
    parser.add_argument("--time-shifting-factor", type=float)
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Parse the layout and exit without loading model weights",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def default_output_dir() -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return ROOT / "runs/showo2_guided" / f"guided-{timestamp}"


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.vae_path = args.vae_path.resolve()
    args.output_dir = (args.output_dir or default_output_dir()).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from guidance import (
        EnergyConfig,
        GuidedEulerSampler,
        GuiderConfig,
        LatentLayoutGuider,
        LayoutEnergy,
        layout_to_dict,
        load_layout_json,
        map_layout_tokens,
        parse_position_layout,
    )

    if args.layout_json:
        layout = load_layout_json(args.prompt, args.layout_json)
    else:
        layout = parse_position_layout(
            args.prompt,
            margin=args.box_margin,
            gap=args.box_gap,
        )

    if args.parse_only:
        payload = layout_to_dict(layout)
        write_json(args.output_dir / "layout.json", payload)
        print(json.dumps({"status": "ok", "layout": payload}, ensure_ascii=False))
        return

    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.vae_path.is_file():
        raise FileNotFoundError(f"Wan VAE checkpoint not found: {args.vae_path}")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Guided Show-o2 inference currently requires CUDA")

    from omegaconf import OmegaConf
    from torch.nn.attention.flex_attention import create_block_mask

    from models import (
        Showo2Qwen2_5,
        WanVAE,
        omni_attn_mask,
        omni_attn_mask_naive,
    )
    from models.misc import get_text_tokenizer, prepare_gen_input
    from transport import create_transport
    from utils import denorm, get_hyper_params, path_to_llm_name

    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)
    weight_type = dtype_from_name(args.weight_type)
    vae_weight_type = dtype_from_name(args.vae_weight_type)
    device = torch.device(args.device)

    seed_everything(args.seed)
    tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
    )
    layout = map_layout_tokens(layout, tokenizer)
    write_json(args.output_dir / "layout.json", layout_to_dict(layout))

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
        default_cfg_scale,
    ) = get_hyper_params(config, tokenizer, showo_token_ids)

    guidance_scale = (
        float(args.guidance_scale)
        if args.guidance_scale is not None
        else float(default_cfg_scale)
    )
    num_inference_steps = (
        int(args.num_inference_steps)
        if args.num_inference_steps is not None
        else int(config.transport.num_inference_steps)
    )
    time_shifting_factor = (
        float(args.time_shifting_factor)
        if args.time_shifting_factor is not None
        else (
            float(config.transport.time_shifting_factor)
            if config.transport.time_shifting_factor is not None
            else None
        )
    )
    if args.guidance_steps > num_inference_steps - 1:
        raise ValueError("guidance_steps cannot exceed the number of Euler intervals")
    if args.backtracking_steps > 0 and args.sampling_attention_mask != "flex":
        raise ValueError("backtracking guidance requires --sampling-attention-mask flex")

    print(json.dumps({"status": "loading", "model": args.model}, sort_keys=True))
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
    vae = WanVAE(
        vae_pth=str(args.vae_path),
        dtype=vae_weight_type,
        device=device,
    )

    (
        text_tokens_cond,
        text_tokens_null,
        positions_cond,
        positions_null,
    ) = prepare_gen_input(
        [args.prompt],
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

    latent_shape = (
        1,
        image_latent_dim,
        latent_height * patch_size,
        latent_width * patch_size,
    )
    initial_latents = torch.randn(
        latent_shape,
        device=device,
        dtype=weight_type,
    )
    if guidance_scale > 0.0:
        initial_latents = torch.cat([initial_latents, initial_latents], dim=0)
        text_tokens = torch.cat([text_tokens_cond, text_tokens_null], dim=0)
        modality_positions = torch.cat([positions_cond, positions_null], dim=0)
    else:
        text_tokens = text_tokens_cond
        modality_positions = positions_cond

    attention_mask = omni_attn_mask_naive(
        text_tokens.size(0),
        max_seq_len,
        modality_positions,
        device,
    ).to(weight_type)
    model_kwargs = {
        "text_tokens": text_tokens,
        "attention_mask": attention_mask,
        "modality_positions": modality_positions,
        "output_hidden_states": True,
        "max_seq_len": max_seq_len,
        "guidance_scale": guidance_scale,
    }
    velocity_model_kwargs = model_kwargs
    if args.sampling_attention_mask == "flex":
        velocity_model_kwargs = dict(model_kwargs)
        velocity_model_kwargs["attention_mask"] = create_block_mask(
            omni_attn_mask(modality_positions),
            B=text_tokens.size(0),
            H=None,
            Q_LEN=max_seq_len,
            KV_LEN=max_seq_len,
            device=device,
        )

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
    energy = LayoutEnergy(
        EnergyConfig(
            layers=args.layers,
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
        layouts=(layout,),
        modality_positions=positions_cond,
        spatial_shape=(latent_height, latent_width),
        add_time_embeds=add_time_embeds,
        config=GuiderConfig(
            inner_steps=args.inner_steps,
            step_size=args.energy_step_size,
            max_grad_rms=args.max_grad_rms,
            max_flow_step_ratio=args.max_flow_step_ratio,
            timestep_decay_power=args.timestep_decay_power,
            attention_source=args.attention_source,
            backtracking_steps=args.backtracking_steps,
            backtracking_factor=args.backtracking_factor,
            min_relative_drop=args.min_relative_drop,
            lookahead_weight=args.lookahead_weight,
            presence_preservation_ratio=args.presence_preservation_ratio,
        ),
    )
    sampler = GuidedEulerSampler(
        transport=transport,
        num_steps=num_inference_steps,
        guidance_steps=args.guidance_steps,
        time_shifting_factor=time_shifting_factor,
        reverse=bool(config.transport.reverse),
    )

    def report(record: dict[str, Any]) -> None:
        energy_value = None
        if record["guidance"]:
            energy_value = record["guidance"][-1]["energy"]
        print(
            json.dumps(
                {
                    "step": record["step"] + 1,
                    "total": num_inference_steps - 1,
                    "time": round(record["time"], 6),
                    "guided": record["guided"],
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
    if guidance_scale > 0.0:
        final_latents = torch.chunk(final_latents, 2, dim=0)[0]

    with torch.no_grad():
        decoded = vae.batch_decode(final_latents.unsqueeze(2)).squeeze(2)
    image_array = denorm(decoded)[0]
    image_path = args.output_dir / "guided.png"
    Image.fromarray(image_array).save(image_path)

    write_json(args.output_dir / "diagnostics.json", diagnostics)
    write_json(
        args.output_dir / "run.json",
        {
            "status": "completed",
            "model": args.model,
            "config": str(args.config),
            "vae_path": str(args.vae_path),
            "prompt": args.prompt,
            "seed": args.seed,
            "image": str(image_path),
            "guidance": {
                "cfg_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "guidance_steps": args.guidance_steps,
                "inner_steps": args.inner_steps,
                "energy_step_size": args.energy_step_size,
                "max_grad_rms": args.max_grad_rms,
                "max_flow_step_ratio": args.max_flow_step_ratio,
                "timestep_decay_power": args.timestep_decay_power,
                "backtracking_steps": args.backtracking_steps,
                "backtracking_factor": args.backtracking_factor,
                "min_relative_drop": args.min_relative_drop,
                "lookahead_weight": args.lookahead_weight,
                "presence_preservation_ratio": args.presence_preservation_ratio,
                "sampling_attention_mask": args.sampling_attention_mask,
                "layers": list(args.layers),
                "attention_source": args.attention_source,
                "inside_weight": args.inside_weight,
                "cross_weight": args.cross_weight,
                "time_shifting_factor": time_shifting_factor,
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
    print(
        json.dumps(
            {
                "status": "completed",
                "image": str(image_path),
                "diagnostics": str(args.output_dir / "diagnostics.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
