#!/usr/bin/env python3
"""Score generated SR2D images with the official VISOR OWL-ViT protocol."""

from __future__ import annotations

import argparse
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

from evaluation.c8_self_cycle.run_self_cycle import file_sha256
from evaluation.c9_visor.generate_images import image_path, read_manifest
from evaluation.c9_visor.visor_metrics import aggregate_visor, score_detections




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--generation-run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--owlvit-model",
        type=Path,
        default=WORKSPACE_ROOT / "models/visor/owlvit-base-patch32",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_results(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (int(row["unique_id"]), int(row["sample_index"]))
        if key in rows:
            raise ValueError(f"duplicate VISOR result at {path}:{line_number}")
        rows[key] = row
    return rows


def process_owlvit_output(
    output: dict[str, torch.Tensor],
    objects: tuple[str, str],
    threshold: float,
) -> tuple[list[str], list[list[float]], list[list[float]], list[float]]:
    labels: list[str] = []
    centroids: list[list[float]] = []
    boxes: list[list[float]] = []
    scores: list[float] = []
    for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
        value = float(score.detach().cpu())
        if value <= threshold:
            continue
        label_index = int(label.detach().cpu())
        if label_index < 0 or label_index >= len(objects):
            raise ValueError(f"OWL-ViT returned invalid query label {label_index}")
        coordinates = [float(item) for item in box.detach().cpu().tolist()]
        labels.append(objects[label_index])
        boxes.append(coordinates)
        scores.append(value)
        centroids.append(
            [
                0.5 * (coordinates[0] + coordinates[2]),
                0.5 * (coordinates[1] + coordinates[3]),
            ]
        )
    return labels, centroids, boxes, scores


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("OWL-ViT threshold must be in [0, 1]")
    if args.samples_per_prompt < 1:
        raise ValueError("samples-per-prompt must be positive")
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
    missing_images = [
        str(image_path(args.generation_run_dir, int(record["unique_id"]), sample_index))
        for record, sample_index in tasks
        if not image_path(
            args.generation_run_dir, int(record["unique_id"]), sample_index
        ).is_file()
    ]
    if missing_images and not args.dry_run:
        raise FileNotFoundError(
            f"VISOR scoring is missing {len(missing_images)} images; first={missing_images[0]}"
        )
    safetensors_path = args.owlvit_model / "model.safetensors"
    if not args.dry_run and not safetensors_path.is_file():
        raise FileNotFoundError(safetensors_path)
    protocol = {
        "protocol_version": "sr2d-visor-owlvit-official-v1",
        "official_detector": "google/owlvit-base-patch32",
        "owlvit_model": str(args.owlvit_model.resolve()),
        "owlvit_safetensors_sha256": (
            file_sha256(safetensors_path) if safetensors_path.is_file() else None
        ),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "generation_run_dir": str(args.generation_run_dir.resolve()),
        "threshold": args.threshold,
        "query_template": "a photo of a {object}",
        "geometry": "official-any-box-pair-centroid-sign",
        "samples_per_prompt": args.samples_per_prompt,
        "limit_prompts": args.limit_prompts,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.run_dir / "protocol.json"
    if protocol_path.is_file():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("refusing to resume VISOR scorer: protocol differs")
    else:
        write_json(protocol_path, protocol)
    status = {
        "status": "completed" if args.dry_run else "running",
        "phase": "dry_run" if args.dry_run else "object_detection",
        "total_images": len(tasks),
        "completed_images": 0,
        "updated_at": utc_now(),
    }
    write_json(args.run_dir / "status.json", status)
    if args.dry_run:
        summary = {"status": "completed", "dry_run": True, "protocol": protocol}
        write_json(args.run_dir / "summary.json", summary)
        return summary
    if not args.owlvit_model.is_dir():
        raise FileNotFoundError(args.owlvit_model)

    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    device = torch.device(args.device)
    processor = OwlViTProcessor.from_pretrained(
        args.owlvit_model, local_files_only=True
    )
    model = (
        OwlViTForObjectDetection.from_pretrained(
            args.owlvit_model, local_files_only=True, use_safetensors=True
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    results_path = args.run_dir / "image_results.jsonl"
    completed = load_results(results_path)
    status.update(completed_images=len(completed))
    write_json(args.run_dir / "status.json", status)
    for record, sample_index in tasks:
        unique_id = int(record["unique_id"])
        key = (unique_id, sample_index)
        if key in completed:
            continue
        path = image_path(args.generation_run_dir, unique_id, sample_index)
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        objects = (str(record["object_a"]), str(record["object_b"]))
        queries = [[f"a photo of a {objects[0]}", f"a photo of a {objects[1]}"]]
        inputs = processor(text=queries, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([[image.height, image.width]], device=device)
        postprocessed = processor.post_process_object_detection(
            outputs=outputs, threshold=0.0, target_sizes=target_sizes
        )[0]
        labels, centroids, boxes, scores = process_owlvit_output(
            postprocessed, objects, args.threshold
        )
        decision = score_detections(
            labels,
            centroids,
            objects[0],
            objects[1],
            str(record["relation"]),
        )
        row = {
            "unique_id": unique_id,
            "sample_index": sample_index,
            "prompt": str(record["prompt"]),
            "object_a": objects[0],
            "object_b": objects[1],
            "relation": str(record["relation"]),
            "image_path": str(path.resolve()),
            "classes": labels,
            "centroids": centroids,
            "boxes_xyxy": boxes,
            "scores": scores,
            **decision,
        }
        append_jsonl(results_path, row)
        completed[key] = row
        status.update(completed_images=len(completed), updated_at=utc_now())
        write_json(args.run_dir / "status.json", status)
        del inputs, outputs, postprocessed

    ordered = [completed[key] for key in sorted(completed)]
    expected = {(int(record["unique_id"]), sample) for record, sample in tasks}
    if set(completed) != expected:
        raise RuntimeError("VISOR result set does not match the requested image tasks")
    metrics = aggregate_visor(ordered)
    official_results = {
        f"{row['unique_id']}_{row['sample_index']}": {
            "classes": row["classes"],
            "centroid": row["centroids"],
            "recall": row["recall"],
            "sra": int(bool(row["spatial_correct"])),
            "text": row["prompt"],
        }
        for row in ordered
    }
    write_json(args.run_dir / "official_detection_results.json", official_results)
    summary = {
        "status": "completed",
        "completed_at": utc_now(),
        "protocol": protocol,
        "metrics": metrics,
        "image_results": str(results_path.resolve()),
    }
    write_json(args.run_dir / "summary.json", summary)
    status.update(status="completed", phase="completed", updated_at=utc_now())
    write_json(args.run_dir / "status.json", status)
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
