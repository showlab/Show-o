#!/usr/bin/env python3
"""Run B-prime position training with the GenEval-aligned v2 objective."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.c5_boxdiff_adapter import bprime_geneval_loss
from evaluation.c5_boxdiff_adapter import train_bprime_position as trainer


def argument_path(name: str) -> Path | None:
    try:
        return Path(sys.argv[sys.argv.index(name) + 1]).expanduser().resolve()
    except (ValueError, IndexError):
        return None


def main() -> int:
    # Keep the proven model/data/checkpoint path while replacing every loss
    # entry point as one coherent protocol. The legacy module remains untouched.
    trainer.LOSS_NAMES = bprime_geneval_loss.LOSS_NAMES
    trainer.position_only_losses = bprime_geneval_loss.position_only_losses
    trainer.calibrate_weights = bprime_geneval_loss.calibrate_weights
    trainer.compose_position_loss = bprime_geneval_loss.compose_position_loss
    result = trainer.main()
    run_dir = argument_path("--run-dir")
    metadata_path = run_dir / "metadata.json" if run_dir is not None else None
    if result == 0 and metadata_path is not None and metadata_path.is_file():
        payload = json.loads(metadata_path.read_text())
        payload.update({
            "protocol": "bprime-position-only-geneval-surrogate-v2",
            "benchmark_alignment": {
                "evaluator": "GenEval relative_position",
                "position_threshold": 0.1,
                "relation_threshold": 0.5,
                "training_margin": 0.1,
                "absolute_box_regression": False,
            },
        })
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(metadata_path)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
