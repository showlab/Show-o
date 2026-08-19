#!/usr/bin/env python3
"""Compute official Google Research CMMD with reusable embedding caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    directory_fingerprint,
    read_coco_manifest,
    utc_now,
    validate_flat_image_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--google-research-commit", required=True)
    parser.add_argument("--scenic-commit", required=True)
    return parser.parse_args()


def parse_variants(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"variant must be NAME=PATH: {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        path = Path(raw_path).resolve()
        if not name or name in result:
            raise ValueError(f"invalid or duplicate variant: {name!r}")
        result[name] = path
    return result


def cache_key(fingerprint: str, count: int, commit: str, scenic: str) -> str:
    value = f"official-cmmd\0{commit}\0{scenic}\0{count}\0{fingerprint}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_or_compute_embeddings(
    name: str,
    root: Path,
    filenames: list[str],
    cache_dir: Path,
    embedding_model: Any,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    from cmmd import io_util

    fingerprint, total_bytes = directory_fingerprint(root, filenames)
    key = cache_key(
        fingerprint,
        len(filenames),
        args.google_research_commit,
        args.scenic_commit,
    )
    path = cache_dir / f"{name}-{key}.npz"
    if path.is_file():
        payload = np.load(path)
        embeddings = np.asarray(payload["embeddings"])
        if embeddings.shape != (len(filenames), 768):
            raise RuntimeError(f"invalid CMMD cache shape in {path}")
        return embeddings, {
            "cache_hit": True,
            "cache_path": str(path.resolve()),
            "directory_fingerprint": fingerprint,
            "directory_bytes": total_bytes,
        }
    embeddings = np.asarray(
        io_util.compute_embeddings_for_dir(
            str(root), embedding_model, args.batch_size, len(filenames)
        )
    )
    if embeddings.shape != (len(filenames), 768):
        raise RuntimeError(
            f"expected {(len(filenames), 768)} embeddings for {name}, "
            f"got {embeddings.shape}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, embeddings=embeddings)
    os.replace(temporary, path)
    return embeddings, {
        "cache_hit": False,
        "cache_path": str(path.resolve()),
        "directory_fingerprint": fingerprint,
        "directory_bytes": total_bytes,
    }


def main() -> None:
    args = parse_args()
    rows = read_coco_manifest(args.manifest)
    if args.batch_size < 1 or len(rows) % args.batch_size:
        raise ValueError(
            "official CMMD drops incomplete batches; choose a batch size that "
            "exactly divides the manifest size"
        )
    variants = parse_variants(args.variant)
    reference_filenames = [str(row["reference_filename"]) for row in rows]
    generated_filenames = [f"{index:05d}.png" for index in range(len(rows))]
    validate_flat_image_dir(args.reference_dir, reference_filenames)
    for root in variants.values():
        validate_flat_image_dir(root, generated_filenames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    roots = {"reference": args.reference_dir, **variants}
    write_json(
        status_path,
        {
            "status": "running",
            "phase": "loading_scenic_clip",
            "variants": list(variants),
            "total_embedding_sets": len(roots),
            "completed_embedding_sets": 0,
            "percent": 0.0,
            "updated_at": utc_now(),
        },
    )
    started = time.perf_counter()
    try:
        import flax
        import jax
        import tensorflow as tf
        from cmmd import distance, embedding

        embedding_model = embedding.ClipEmbeddingModel()
        filenames_by_name = {
            "reference": reference_filenames,
            **{name: generated_filenames for name in variants},
        }
        embedding_sets: dict[str, np.ndarray] = {}
        cache_metadata: dict[str, Any] = {}
        for completed, (name, root) in enumerate(roots.items(), 1):
            write_json(
                status_path,
                {
                    "status": "running",
                    "phase": "clip_embeddings",
                    "current": name,
                    "variants": list(variants),
                    "total_embedding_sets": len(roots),
                    "completed_embedding_sets": completed - 1,
                    "percent": 100.0 * (completed - 1) / len(roots),
                    "updated_at": utc_now(),
                },
            )
            embedding_sets[name], cache_metadata[name] = load_or_compute_embeddings(
                name,
                root,
                filenames_by_name[name],
                args.embedding_cache_dir,
                embedding_model,
                args,
            )
        reference = embedding_sets["reference"]
        scores = {
            name: {
                "cmmd": float(np.asarray(distance.mmd(reference, embedding_sets[name]))),
                "images": len(rows),
            }
            for name in variants
        }
        summary = {
            "status": "completed",
            "protocol": "showo2-paired-coco-official-cmmd-v2",
            "manifest": str(args.manifest.resolve()),
            "images_per_set": len(rows),
            "implementation": {
                "source": "google-research/cmmd",
                "google_research_commit": args.google_research_commit,
                "scenic_commit": args.scenic_commit,
                "embedding_model": "OpenAI CLIP ViT-L/14@336px via Scenic",
                "image_preprocess": "official CMMD center-crop, bicubic resize, CLIP normalize",
                "mmd": "official biased/minimum-variance Gaussian RBF estimator",
                "sigma": 10,
                "scale": 1000,
                "batch_size": args.batch_size,
                "drop_remainder": False,
                "drop_remainder_note": (
                    "Official code uses drop_remainder=True; batch size exactly "
                    f"divides {len(rows)} so zero images are dropped."
                ),
            },
            "scores": scores,
            "embedding_cache": cache_metadata,
            "environment": {
                "python": platform.python_version(),
                "jax": jax.__version__,
                "flax": flax.__version__,
                "tensorflow": tf.__version__,
                "jax_devices": [str(device) for device in jax.devices()],
            },
            "runtime_seconds": time.perf_counter() - started,
            "updated_at": utc_now(),
        }
        write_json(args.output_dir / "summary.json", summary)
        write_json(
            status_path,
            {
                "status": "completed",
                "phase": "completed",
                "variants": list(variants),
                "total_embedding_sets": len(roots),
                "completed_embedding_sets": len(roots),
                "percent": 100.0,
                "updated_at": utc_now(),
            },
        )
        (args.output_dir / "SCORING_COMPLETED").touch()
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    except BaseException as error:
        write_json(
            status_path,
            {
                "status": "failed",
                "phase": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "updated_at": utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    main()
