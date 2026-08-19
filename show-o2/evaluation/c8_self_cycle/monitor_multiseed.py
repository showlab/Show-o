#!/usr/bin/env python3
"""Display queue-level progress for paired C8 self-cycle multiseed runs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_VARIANTS = ("baseline", "mask_logits_500_prompt_conditioned")
MODEL_LABELS = {
    "baseline": "baseline",
    "mask_logits_500_prompt_conditioned": "Mask-logits500",
}


@dataclass(frozen=True)
class RunProgress:
    seed: int
    variant: str
    run_dir: Path
    state: str
    phase: str
    percent: float
    generation_percent: float
    qa_percent: float
    completed_images: int
    total_images: int
    completed_questions: int
    total_questions: int
    eta_seconds: float | None
    elapsed_seconds: float | None
    error: str | None


def parse_csv(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result or len(set(result)) != len(result):
        raise ValueError("comma-separated values must be non-empty and unique")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue_dir", type=Path)
    parser.add_argument(
        "--seeds",
        help="Comma-separated seed list; defaults to discovered seed-* directories",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variants in queue order",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    return parser.parse_args()


def discover_seeds(queue_dir: Path) -> list[int]:
    seeds = []
    for path in queue_dir.glob("seed-*"):
        if not path.is_dir():
            continue
        try:
            seeds.append(int(path.name.removeprefix("seed-")))
        except ValueError:
            continue
    if not seeds:
        raise FileNotFoundError(f"no seed-* directories found under {queue_dir}")
    return sorted(set(seeds))


def duration(value: float | None) -> str:
    if value is None:
        return "unknown"
    seconds = max(0, int(value))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def bar(percent: float, width: int = 28) -> str:
    filled = min(width, max(0, round(width * percent / 100.0)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def read_progress(queue_dir: Path, seed: int, variant: str) -> RunProgress:
    run_dir = queue_dir / f"seed-{seed}" / variant
    status_path = run_dir / "status.json"
    data = {}
    if status_path.is_file():
        data = json.loads(status_path.read_text(encoding="utf-8"))

    if (run_dir / "BENCHMARK_FAILED").is_file() or data.get("status") == "failed":
        state = "failed"
    elif (run_dir / "BENCHMARK_COMPLETED").is_file() or data.get(
        "status"
    ) == "completed":
        state = "completed"
    elif status_path.is_file():
        state = "running"
    else:
        state = "pending"

    percent = 100.0 if state == "completed" else float(data.get("percent", 0.0))
    return RunProgress(
        seed=seed,
        variant=variant,
        run_dir=run_dir,
        state=state,
        phase=str(data.get("phase", "not_started")),
        percent=percent,
        generation_percent=float(data.get("generation_percent", 0.0)),
        qa_percent=float(data.get("qa_percent", 0.0)),
        completed_images=int(data.get("completed_images", 0)),
        total_images=int(data.get("total_images", 0)),
        completed_questions=int(data.get("completed_questions", 0)),
        total_questions=int(data.get("total_questions", 0)),
        eta_seconds=(
            float(data["eta_seconds"]) if data.get("eta_seconds") is not None else None
        ),
        elapsed_seconds=(
            float(data["elapsed_seconds"])
            if data.get("elapsed_seconds") is not None
            else None
        ),
        error=str(data["error"]) if data.get("error") else None,
    )


def queue_eta(runs: list[RunProgress], active_index: int | None) -> float | None:
    completed_times = [
        run.elapsed_seconds
        for run in runs
        if run.state == "completed" and run.elapsed_seconds is not None
    ]
    typical = statistics.median(completed_times) if completed_times else None
    if active_index is None:
        pending = sum(run.state == "pending" for run in runs)
        return typical * pending if typical is not None else None
    active = runs[active_index]
    remaining_runs = len(runs) - active_index - 1
    if typical is None:
        return active.eta_seconds
    return (active.eta_seconds or 0.0) + typical * remaining_runs


def render(
    queue_dir: Path,
    seeds: list[int],
    variants: list[str],
) -> tuple[str, str]:
    runs = [
        read_progress(queue_dir, seed, variant)
        for seed in seeds
        for variant in variants
    ]
    completed = sum(run.state == "completed" for run in runs)
    failed = sum(run.state == "failed" for run in runs)
    active_index = next(
        (index for index, run in enumerate(runs) if run.state == "running"),
        None,
    )
    overall = sum(run.percent for run in runs) / len(runs)

    if failed:
        queue_state = "failed"
    elif completed == len(runs) or (queue_dir / "QUEUE_COMPLETED").is_file():
        queue_state = "completed"
    elif active_index is not None:
        queue_state = "running"
    else:
        queue_state = "pending"

    current_text = "current=none"
    detail_lines = []
    if active_index is not None:
        active = runs[active_index]
        model = MODEL_LABELS.get(active.variant, active.variant)
        current_text = (
            f"current={active_index + 1}/{len(runs)} "
            f"model={model} seed={active.seed} phase={active.phase}"
        )
        detail_lines = [
            f"current run {bar(active.percent)} {active.percent:6.2f}% "
            f"eta={duration(active.eta_seconds)}",
            f"generation  {bar(active.generation_percent)} "
            f"{active.generation_percent:6.2f}% "
            f"images={active.completed_images}/{active.total_images}",
            f"structured  {bar(active.qa_percent)} {active.qa_percent:6.2f}% "
            f"questions={active.completed_questions}/{active.total_questions}",
        ]

    lines = [
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"queue={queue_dir.name} status={queue_state}",
        f"queue total {bar(overall)} {overall:6.2f}% "
        f"completed={completed}/{len(runs)} failed={failed} "
        f"eta~={duration(queue_eta(runs, active_index))}",
        current_text,
        *detail_lines,
        "",
        "runs:",
    ]
    tags = {
        "completed": "DONE",
        "running": "RUN ",
        "failed": "FAIL",
        "pending": "WAIT",
    }
    for index, run in enumerate(runs, 1):
        model = MODEL_LABELS.get(run.variant, run.variant)
        lines.append(
            f"[{tags[run.state]}] {index:>2}/{len(runs)} "
            f"seed={run.seed} model={model:<14} "
            f"progress={run.percent:6.2f}% phase={run.phase}"
        )
        if run.error:
            lines.append(f"       error={run.error}")
    return "\n".join(lines), queue_state


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("interval must be positive")
    queue_dir = args.queue_dir.resolve()
    if not queue_dir.is_dir():
        raise FileNotFoundError(queue_dir)
    seeds = (
        [int(value) for value in parse_csv(args.seeds)]
        if args.seeds
        else discover_seeds(queue_dir)
    )
    variants = parse_csv(args.variants)

    while True:
        output, state = render(queue_dir, seeds, variants)
        if args.watch and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print(output, flush=True)
        if not args.watch or state in {"completed", "failed"}:
            return 1 if state == "failed" else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
