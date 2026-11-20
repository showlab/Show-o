"""V2 batch inference: final + DeepStack fusion, with resumable output."""

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import train_qwen3_showo2_adapter as base

# Preserve the existing input preparation and output formatting.
from inference_qwen3_showo2_fusion_ablation_batch import (
    prepare_qwen_inputs,
    move_inputs_to_device,
    resolve_images,
    resolve_prompts,
    resolve_stages,
    sanitize_name,
    save_result,
)

# RAG_INTEGRATION_V1: retrieval is imported lazily only when enabled.
from investigative_rag.inference_support import OBSERVATION_PROMPT, final_prompt

from qwen3_showo2_deepstack import (
    DeepStackAdapter,
    load_adapter,
    unpack_visual,
    fuse_features,
    patched_image_features,
)


# ============================================================
# CONFIG — edit here; no command-line arguments required
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

STAGES = ["stage1", "stage2"]
# Other options:
# STAGES = ["stage2"]
# STAGES = ["pre_stage1", "stage1", "stage2"]

STAGE1_CHECKPOINT = (
    SCRIPT_DIR / "showo2_qwen3_adapter_stage1_v2/checkpoint_best.pt"
)

STAGE2_CHECKPOINT = (
    SCRIPT_DIR / "showo2_qwen3_adapter_stage2_v2/checkpoint_best.pt"
)

IMAGE_PATH = (
     "/home/chatziko/PycharmProjects/PythonProject/IDMVAE/archive/UCA_image_dataset/one_frame_per_video_split/images"
)

PROMPT_PATH = (
     "/home/chatziko/PycharmProjects/PythonProject/CMAlign/api"
)

OUTPUT_PATH = SCRIPT_DIR / "results/ablation_results"

ALPHAS = [i / 10 for i in range(11)]
MAX_NEW_TOKENS = 512
OVERWRITE_EXISTING = False
USE_RAG = False
RAG_TOP_K = 5
RAG_MIN_SCORE = None  # Do not use an uncalibrated similarity cutoff.
RAG_DIR = SCRIPT_DIR / "investigative_rag"
RAG_OBSERVATION_MAX_TOKENS = 180
SEED = 42

# Increment if you later change inference behavior and want fresh results.
RUN_SCHEMA = 1

USE_RAG = False
RAG_TOP_K = 5
RAG_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# UTILITIES
# ============================================================

def reset_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def file_hash(path):
    """Hash file contents so changed checkpoints/images trigger fresh results."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_hash(identity):
    text = json.dumps(identity, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def result_paths(output, image, prompt, stage, identity, single_run):
    output = Path(output).expanduser()

    # Preserve the old single-combination behavior:
    # an explicit .json output path is used exactly as supplied.
    if single_run and output.suffix.lower() == ".json":
        return output, output.with_suffix(".txt")

    directory = output / stage
    name = (
        f"{sanitize_name(image.stem)}__"
        f"{sanitize_name(prompt.stem)}__"
        f"{identity_hash(identity)}"
    )
    return directory / f"{name}.json", directory / f"{name}.txt"


def result_is_complete(json_path, txt_path, identity, alphas):
    if not json_path.is_file() or not txt_path.is_file():
        return False

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if data.get("run_identity") != identity:
            return False

        results = data.get("results")
        if not isinstance(results, list) or len(results) != len(alphas):
            return False

        return all(
            isinstance(item, dict)
            and item.get("alpha_qwen") == alpha
            and isinstance(item.get("output"), str)
            for item, alpha in zip(results, alphas)
        )
    except (OSError, ValueError, TypeError):
        return False


def save_v2_result(job, image_path, checkpoint_path, norms, results):
    # Keep the existing JSON + TXT formatting.
    save_result(
        json_path=job["json_path"],
        txt_path=job["txt_path"],
        image_path=image_path,
        stage=job["stage"],
        checkpoint_path=checkpoint_path,
        prompt_path=job["prompt_path"],
        prompt=job["prompt"],
        qwen_norm=norms[0],
        showo_norm=norms[1],
        results=results,
    )

    # Add V2 metadata and resume identity.
    path = job["json_path"]
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(
        fusion_version=2,
        fusion_scope="final_and_deepstack",
        run_identity=job["identity"],
        **({"rag_enabled": True, "rag_provenance": job["identity"].get("rag")}
           if job["identity"].get("rag") else {}),
    )

    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)

    with job["txt_path"].open("a", encoding="utf-8") as handle:
        handle.write(
            "\nFusion version: 2\n"
            "Fusion scope: final visual features and all DeepStack levels\n"
        )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

@torch.no_grad()
def process_image(image_path, jobs, checkpoints, args):
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")

    active_stages = list(dict.fromkeys(job["stage"] for job in jobs))
    raw_by_stage = {}

    # Extract Show-o features before loading the full Qwen model.
    for stage in active_stages:
        reset_seed()
        showo, vae = base.load_showo2_visual()

        checkpoint = checkpoints[stage]["data"]
        if checkpoint is not None:
            showo.fusion_proj.load_state_dict(
                checkpoint["fusion_proj"], strict=True
            )

        showo.eval()
        raw_by_stage[stage] = base.get_showo2_features(
            [image],
            showo,
            vae,
            deterministic_vae=True,
        ).detach().cpu()

        del showo, vae
        cleanup()

    processor = AutoProcessor.from_pretrained(base.QWEN3_MODEL)
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        base.QWEN3_MODEL,
        dtype=base.DTYPE,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(base.DEVICE).eval()

    first_inputs = move_inputs_to_device(
        prepare_qwen_inputs(processor, image, jobs[0]["prompt"]),
        base.DEVICE,
    )

    embeddings, deep = unpack_visual(
        qwen.model.get_image_features(
            pixel_values=first_inputs["pixel_values"],
            image_grid_thw=first_inputs["image_grid_thw"],
        )
    )

    if not isinstance(embeddings, (tuple, list)) or len(embeddings) != 1:
        raise ValueError(
            "Expected one image's split embeddings "
            "(the Transformers 4.57.6 API)."
        )

    native = embeddings[0], deep
    indexes = list(qwen.model.visual.deepstack_visual_indexes)

    del first_inputs
    ready = {}

    for stage in active_stages:
        checkpoint = checkpoints[stage]["data"]
        raw = raw_by_stage.pop(stage).to(native[0])

        config = {
            "showo_dim": raw.shape[-1],
            "qwen_dim": native[0].shape[-1],
            "levels": len(deep),
        }

        if checkpoint is None:
            reset_seed()
            adapter = DeepStackAdapter(**config).to(
                device=base.DEVICE,
                dtype=base.DTYPE,
            )
        else:
            if checkpoint["adapter_config"] != config:
                raise ValueError(f"{stage}: adapter dimensions differ")
            if checkpoint["deepstack_indexes"] != indexes:
                raise ValueError(f"{stage}: DeepStack indexes differ")

            adapter = load_adapter(
                checkpoint,
                device=base.DEVICE,
                dtype=base.DTYPE,
            )

        adapter.eval()

        aligned = base.spatial_align_showo_to_qwen(
            raw, native[0].unsqueeze(0)
        )
        final, predicted_deep = adapter(aligned)
        adapted = final[0], [value[0] for value in predicted_deep]

        # Validate final and intermediate shapes before generation.
        fuse_features(native, adapted, 0.5)

        qwen_norm = native[0].float().norm(dim=-1).mean().item()
        showo_norm = adapted[0].float().norm(dim=-1).mean().item()

        ready[stage] = {
            "adapted": adapted,
            "norms": (qwen_norm, showo_norm),
        }

        del adapter, aligned, raw, final, predicted_deep, adapted

    cleanup()

    for job_index, job in enumerate(jobs, start=1):
        stage = job["stage"]
        stage_ready = ready[stage]
        qwen_norm, showo_norm = stage_ready["norms"]

        print(
            f"\nJob {job_index}/{len(jobs)} | "
            f"{image_path.name} | {stage} | "
            f"{job['prompt_path'].name}",
            flush=True,
        )

        inputs = move_inputs_to_device(
            prepare_qwen_inputs(processor, image, job["prompt"]),
            base.DEVICE,
        )

        results = []

        for alpha in args.alphas:
            reset_seed()

            # The same alpha applies to final and intermediate features.
            final, fused_deep = fuse_features(
                native,
                stage_ready["adapted"],
                alpha,
            )

            rag_observation = None
            retrieved_knowledge = []
            effective_inputs = inputs
            effective_prompt = job["prompt"]

            if args.use_rag:
                # Two-pass mode: obtain *neutral* observations before retrieval.
                observation_inputs = move_inputs_to_device(
                    prepare_qwen_inputs(processor, image, OBSERVATION_PROMPT),
                    base.DEVICE,
                )
                reset_seed()
                with patched_image_features(qwen, final, fused_deep):
                    observation_ids = qwen.generate(
                        **observation_inputs,
                        max_new_tokens=args.rag_observation_max_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                rag_observation = processor.batch_decode(
                    observation_ids[:, observation_inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                from investigative_rag.inference_support import retrieve_context
                context, retrieved_knowledge = retrieve_context(
                    args.rag_retriever, rag_observation,
                    top_k=args.rag_top_k, min_score=args.rag_min_score,
                )
                effective_prompt = final_prompt(job["prompt"], context)
                effective_inputs = move_inputs_to_device(
                    prepare_qwen_inputs(processor, image, effective_prompt),
                    base.DEVICE,
                )

            if args.use_rag:
                reset_seed()
            with patched_image_features(qwen, final, fused_deep):
                ids = qwen.generate(
                    **effective_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

            generated = ids[:, effective_inputs["input_ids"].shape[1]:]
            text = processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            results.append({
                "alpha_qwen": alpha,
                "beta_showo2": 1 - alpha,
                "qwen_norm": qwen_norm,
                "showo2_norm": showo_norm,
                "output": text,
                **({
                    "rag_enabled": True,
                    "rag_observation": rag_observation,
                    "retrieved_knowledge": retrieved_knowledge,
                    "rag_effective_prompt_sha256": hashlib.sha256(
                        effective_prompt.encode("utf-8")
                    ).hexdigest(),
                } if args.use_rag else {}),
            })

            print(f"\nalpha={alpha:.1f}\n{text}", flush=True)
            del ids, generated, final, fused_deep

        save_v2_result(
            job,
            image_path,
            checkpoints[stage]["path"],
            stage_ready["norms"],
            results,
        )
        del inputs

    # GPU tensors are released when this function returns.


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--image", default=IMAGE_PATH)
    parser.add_argument("--prompt-file", default=PROMPT_PATH)
    parser.add_argument("--output", default=str(OUTPUT_PATH))

    parser.add_argument(
        "--stage",
        nargs="+",
        choices=["pre_stage1", "stage1", "stage2", "all"],
        default=STAGES,
    )
    parser.add_argument(
        "--stage1-checkpoint",
        default=str(STAGE1_CHECKPOINT),
    )
    parser.add_argument(
        "--stage2-checkpoint",
        default=str(STAGE2_CHECKPOINT),
    )
    parser.add_argument("--vae-path", default=base.WAN_VAE_PATH)
    parser.add_argument("--use-rag", action="store_true", default=USE_RAG)
    parser.add_argument("--rag-dir", default=str(RAG_DIR))
    parser.add_argument("--rag-top-k", type=int, default=RAG_TOP_K)
    parser.add_argument("--rag-min-score", type=float, default=RAG_MIN_SCORE)
    parser.add_argument("--rag-observation-max-tokens", type=int,
                        default=RAG_OBSERVATION_MAX_TOKENS)
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=ALPHAS
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=MAX_NEW_TOKENS
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        default=OVERWRITE_EXISTING,
    )

    args = parser.parse_args()

    args.rag_retriever = None
    args.rag_provenance = None
    if args.use_rag:
        from investigative_rag.inference_support import initialize_rag
        args.rag_retriever, args.rag_provenance = initialize_rag(
            args.rag_dir, top_k=args.rag_top_k
        )
        if args.rag_observation_max_tokens < 1:
            parser.error("rag-observation-max-tokens must be positive")

    if any(not 0 <= alpha <= 1 for alpha in args.alphas):
        parser.error("alphas must be finite and in [0, 1]")
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")

    base.WAN_VAE_PATH = args.vae_path
    reset_seed()

    stages = resolve_stages(args.stage)
    images = resolve_images(args.image)
    prompts = resolve_prompts(args.prompt_file)

    prompt_texts = {
        path: path.read_text(encoding="utf-8").strip()
        for path in prompts
    }

    single_run = (
        len(images) == 1
        and len(prompts) == 1
        and len(stages) == 1
    )

    checkpoints = {}
    checkpoint_paths = {
        "stage1": args.stage1_checkpoint,
        "stage2": args.stage2_checkpoint,
    }

    # Load small trained components on CPU once per run.
    for stage in stages:
        if stage == "pre_stage1":
            checkpoints[stage] = {
                "data": None,
                "path": None,
                "sha256": None,
            }
            continue

        path = Path(checkpoint_paths[stage]).expanduser().resolve()
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        expected_stage = 1 if stage == "stage1" else 2
        if checkpoint.get("fusion_version") != 2:
            raise ValueError(f"{path}: expected a V2 checkpoint")
        if checkpoint.get("stage") != expected_stage:
            raise ValueError(
                f"{path}: expected stage {expected_stage}, "
                f"found {checkpoint.get('stage')}"
            )
        if checkpoint["qwen_model"] != base.QWEN3_MODEL:
            raise ValueError(f"{path}: Qwen model differs")

        # Optimizer state is unnecessary for inference.
        checkpoint.pop("optimizer", None)
        checkpoint.pop("scaler", None)

        checkpoints[stage] = {
            "data": checkpoint,
            "path": str(path),
            "sha256": file_hash(path),
        }

        print(
            f"{stage}: {path}\n"
            f"Checkpoint epoch: {checkpoint.get('epoch')}",
            flush=True,
        )

    print(
        f"\nFusion V2 | Images: {len(images)} | "
        f"Prompts: {len(prompts)} | Stages: {stages}\n"
        f"Alphas: {args.alphas}\nOutput: {args.output}",
        flush=True,
    )

    # Include inference settings in the resume identity.
    common_identity = {
        "schema": RUN_SCHEMA,
        "fusion_version": 2,
        "fusion_scope": "final_and_deepstack",
        "qwen_model": base.QWEN3_MODEL,
        "showo_model": base.SHOWO2_MODEL,
        "vae_sha256": file_hash(args.vae_path),
        "showo_size": base.SHOWO_SIZE,
        "qwen_size": base.QWEN_SIZE,
        "dtype": str(base.DTYPE),
        "seed": SEED,
        "alphas": args.alphas,
        "max_new_tokens": args.max_new_tokens,
        # RAG settings change the run identity only in RAG mode.
        **({"rag": args.rag_provenance,
            "rag_min_score": args.rag_min_score,
            "rag_observation_max_tokens": args.rag_observation_max_tokens,
            "rag_observation_prompt": OBSERVATION_PROMPT}
           if args.use_rag else {}),
    }

    for image_index, image_path in enumerate(images, start=1):
        print(
            f"\nIMAGE {image_index}/{len(images)}: {image_path}",
            flush=True,
        )

        image_digest = file_hash(image_path)
        jobs = []

        for prompt_path in prompts:
            prompt = prompt_texts[prompt_path]

            for stage in stages:
                identity = {
                    **common_identity,
                    "stage": stage,
                    "checkpoint_sha256": checkpoints[stage]["sha256"],
                    "image": str(image_path),
                    "image_sha256": image_digest,
                    # Prompt location can change without invalidating results.
                    "prompt": prompt,
                }

                json_path, txt_path = result_paths(
                    args.output,
                    image_path,
                    prompt_path,
                    stage,
                    identity,
                    single_run,
                )

                if (
                    not args.overwrite_existing
                    and result_is_complete(
                        json_path, txt_path, identity, args.alphas
                    )
                ):
                    print(f"[SKIP] {json_path}")
                    continue

                jobs.append({
                    "stage": stage,
                    "prompt_path": prompt_path,
                    "prompt": prompt,
                    "identity": identity,
                    "json_path": json_path,
                    "txt_path": txt_path,
                })

        if not jobs:
            print("[SKIP IMAGE] All requested results are complete.")
            continue

        try:
            process_image(image_path, jobs, checkpoints, args)
        finally:
            cleanup()

    print("\nALL REQUESTED RUNS FINISHED")


if __name__ == "__main__":
    main()