#!/usr/bin/env python3
"""Compute cached CleanFID and KID for paired COCO image directories."""

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
import torch
from cleanfid import fid
from scipy import linalg

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
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--inception-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--kid-seed", type=int, default=20260812)
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-max-subset-size", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def frechet_distance_compatible(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """CleanFID's formula with support for SciPy versions without ``disp``."""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    if mu1.shape != mu2.shape or sigma1.shape != sigma2.shape:
        raise ValueError("FID statistics have incompatible shapes")
    difference = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"FID covariance square root is complex: {np.abs(covmean.imag).max()}")
        covmean = covmean.real
    return float(
        difference.dot(difference)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2.0 * np.trace(covmean)
    )


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


def cache_key(fingerprint: str, mode: str, count: int) -> str:
    value = f"cleanfid-0.1.35\0{mode}\0{count}\0{fingerprint}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_or_compute_features(
    name: str,
    root: Path,
    filenames: list[str],
    cache_dir: Path,
    model: Any,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    fingerprint, total_bytes = directory_fingerprint(root, filenames)
    key = cache_key(fingerprint, "clean", len(filenames))
    path = cache_dir / f"{name}-{key}.npz"
    if path.is_file():
        payload = np.load(path)
        features = np.asarray(payload["features"])
        if features.shape != (len(filenames), 2048):
            raise RuntimeError(f"invalid cached feature shape in {path}")
        return features, {
            "cache_hit": True,
            "cache_path": str(path.resolve()),
            "directory_fingerprint": fingerprint,
            "directory_bytes": total_bytes,
        }
    features = fid.get_folder_features(
        str(root),
        model=model,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        mode="clean",
        description=f"Inception {name}: ",
        verbose=True,
    )
    features = np.asarray(features)
    if features.shape != (len(filenames), 2048):
        raise RuntimeError(
            f"expected {(len(filenames), 2048)} features for {name}, got {features.shape}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, features=features)
    os.replace(temporary, path)
    return features, {
        "cache_hit": False,
        "cache_path": str(path.resolve()),
        "directory_fingerprint": fingerprint,
        "directory_bytes": total_bytes,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    rows = read_coco_manifest(args.manifest)
    variants = parse_variants(args.variant)
    reference_filenames = [str(row["reference_filename"]) for row in rows]
    generated_filenames = [f"{index:05d}.png" for index in range(len(rows))]
    validate_flat_image_dir(args.reference_dir, reference_filenames)
    for root in variants.values():
        validate_flat_image_dir(root, generated_filenames)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CleanFID/KID scoring requires CUDA")
    if not args.inception_path.is_file():
        raise FileNotFoundError(args.inception_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.feature_cache_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    write_json(
        status_path,
        {
            "status": "running",
            "phase": "loading_inception",
            "variants": list(variants),
            "total_feature_sets": len(variants) + 1,
            "completed_feature_sets": 0,
            "percent": 0.0,
            "updated_at": utc_now(),
        },
    )
    started = time.perf_counter()
    try:
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from cleanfid.inception_torchscript import InceptionV3W

        inception = InceptionV3W(
            str(args.inception_path.parent), download=False, resize_inside=False
        ).to(torch.device(args.device))
        inception.eval().requires_grad_(False)

        def model(images: torch.Tensor) -> torch.Tensor:
            return inception(images)

        feature_sets: dict[str, np.ndarray] = {}
        cache_metadata: dict[str, Any] = {}
        roots = {"reference": args.reference_dir, **variants}
        filenames_by_name = {
            "reference": reference_filenames,
            **{name: generated_filenames for name in variants},
        }
        for completed, (name, root) in enumerate(roots.items(), 1):
            write_json(
                status_path,
                {
                    "status": "running",
                    "phase": "inception_features",
                    "current": name,
                    "variants": list(variants),
                    "total_feature_sets": len(roots),
                    "completed_feature_sets": completed - 1,
                    "percent": 100.0 * (completed - 1) / len(roots),
                    "updated_at": utc_now(),
                },
            )
            feature_sets[name], cache_metadata[name] = load_or_compute_features(
                name,
                root,
                filenames_by_name[name],
                args.feature_cache_dir,
                model,
                args,
            )
        reference = feature_sets["reference"]
        reference_mu = np.mean(reference, axis=0)
        reference_sigma = np.cov(reference, rowvar=False)
        scores = {}
        for name in variants:
            generated = feature_sets[name]
            generated_mu = np.mean(generated, axis=0)
            generated_sigma = np.cov(generated, rowvar=False)
            fid_score = frechet_distance_compatible(
                reference_mu, reference_sigma, generated_mu, generated_sigma
            )
            np.random.seed(args.kid_seed)
            kid_score = fid.kernel_distance(
                reference,
                generated,
                num_subsets=args.kid_subsets,
                max_subset_size=args.kid_max_subset_size,
            )
            scores[name] = {
                "fid": float(fid_score),
                "kid": float(kid_score),
                "kid_x1000": float(1000.0 * kid_score),
                "images": len(rows),
            }
        summary = {
            "status": "completed",
            "protocol": "showo2-paired-coco-cleanfid-kid-v2",
            "manifest": str(args.manifest.resolve()),
            "images_per_set": len(rows),
            "implementation": {
                "package": "clean-fid==0.1.35",
                "mode": "clean",
                "feature_model": "inception-2015-12-05.pt",
                "inception_path": str(args.inception_path.resolve()),
                "inception_sha256": hashlib.sha256(
                    args.inception_path.read_bytes()
                ).hexdigest(),
                "fid": "Gaussian Frechet distance on cached 2048D features",
                "kid": "CleanFID polynomial-kernel estimator",
                "kid_seed": args.kid_seed,
                "kid_subsets": args.kid_subsets,
                "kid_max_subset_size": args.kid_max_subset_size,
            },
            "scores": scores,
            "feature_cache": cache_metadata,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0),
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
                "total_feature_sets": len(roots),
                "completed_feature_sets": len(roots),
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
