#!/usr/bin/env python3
"""Monitor final LoRA evaluations for the Mask/Flow/Retention factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VARIANTS = ("mask_logits", "mask_flow", "mask_retention", "mask_flow_retention")
GENEVAL_WEIGHT = 420.0
SELF_CYCLE_WEIGHT = 3758.0


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def fraction(status: dict[str, Any], completed: str, total: str) -> float:
    if status.get("status") == "completed":
        return 1.0
    denominator = float(status.get(total, 0) or 0)
    numerator = float(status.get(completed, 0) or 0)
    return min(1.0, max(0.0, numerator / denominator)) if denominator else 0.0


def evaluation_progress(root: Path) -> tuple[str, float, float | None]:
    eval_roots = sorted((root / "lora/evaluation").glob("step-*"))
    if not eval_roots:
        return "waiting", 0.0, None
    eval_root = eval_roots[-1]
    geneval = read_json(eval_root / "geneval_position72/status.json")
    self_paths = sorted(eval_root.glob("self_cycle/seed-*/**/status.json"))
    self_cycle = read_json(self_paths[-1]) if self_paths else {}
    geneval_fraction = fraction(geneval, "completed_images", "total_images")
    if self_cycle.get("phase") == "generation":
        self_fraction = 0.90 * fraction(
            self_cycle, "completed_images", "total_images"
        )
    elif self_cycle.get("phase") in {"qa", "scoring"}:
        self_fraction = 0.90 + 0.10 * fraction(
            self_cycle, "completed_questions", "total_questions"
        )
    else:
        self_fraction = fraction(
            self_cycle, "completed_questions", "total_questions"
        )
    weighted = (
        GENEVAL_WEIGHT * geneval_fraction + SELF_CYCLE_WEIGHT * self_fraction
    ) / (GENEVAL_WEIGHT + SELF_CYCLE_WEIGHT)
    if (root / "lora/ZERO_INIT_EVAL_COMPLETED").is_file():
        return "completed", 1.0, 0.0
    state = "running" if geneval or self_cycle else "waiting"
    eta_values = [
        float(status["eta_seconds"])
        for status in (geneval, self_cycle)
        if status.get("eta_seconds") is not None
    ]
    return state, weighted, sum(eta_values) if eta_values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--pure-root", type=Path, required=True)
    parser.add_argument("--factorial-root", type=Path, required=True)
    args = parser.parse_args()
    roots = {
        "mask_logits": args.pure_root,
        **{
            name: args.factorial_root / name
            for name in VARIANTS
            if name != "mask_logits"
        },
    }
    rows = [(name, *evaluation_progress(root)) for name, root in roots.items()]
    overall = sum(row[2] for row in rows) / len(rows)
    print(
        f"lora_eval [{int(overall * 25) * '#':<25}] {100.0 * overall:6.2f}% "
        f"completed={sum(row[1] == 'completed' for row in rows)}/4"
    )
    for name, state, progress, eta in rows:
        eta_text = f"{eta / 3600:.2f}h" if eta is not None else "n/a"
        print(
            f"{name:<21} status={state:<10} "
            f"progress={100.0 * progress:6.2f}% eta={eta_text}"
        )
    status = args.run_root / "queue.status"
    if status.is_file():
        print(status.read_text(encoding="utf-8").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
