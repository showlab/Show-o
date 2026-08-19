#!/usr/bin/env python3
"""Monitor the three-checkpoint GenEval and Self-cycle sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

STEPS = (500, 1000, 2500)
SEEDS = (20260810, 20260811, 20260812, 20260813)
HISTORICAL_SECONDS = {"geneval": 420.0, "self_cycle": 3758.0}


@dataclass(frozen=True)
class Task:
    step: int
    benchmark: str
    seed: int | None
    state: str
    phase: str
    percent: float
    detail: str
    eta_seconds: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--session", default="showo2_flow_retention_checkpoint_sweep"
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    value = max(0, round(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def bar(percent: float, width: int = 28) -> str:
    filled = min(width, max(0, round(width * percent / 100.0)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def state_for(run_dir: Path, data: dict) -> str:
    if (run_dir / "BENCHMARK_FAILED").is_file() or data.get("status") == "failed":
        return "failed"
    if (run_dir / "BENCHMARK_COMPLETED").is_file() or data.get("status") == "completed":
        return "completed"
    return "running" if data else "pending"


def rate_eta(data: dict, percent: float) -> float | None:
    if data.get("eta_seconds") is not None:
        return float(data["eta_seconds"])
    started = parse_time(data.get("started_at"))
    if started is None or not 0.0 < percent < 100.0:
        return None
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return elapsed * (100.0 - percent) / percent


def geneval_task(run_root: Path, step: int) -> Task:
    run_dir = run_root / f"step-{step}" / "geneval_position72"
    data = load_json(run_dir / "status.json")
    state = state_for(run_dir, data)
    phase = str(data.get("phase", "not_started"))
    done = int(data.get("completed_images", 0))
    total = int(data.get("total_images", 288)) or 288
    if state == "completed":
        percent = 100.0
    elif phase.startswith("scoring"):
        percent = 95.0
    elif phase == "releasing_generation_models":
        percent = 92.0
    elif phase == "generation":
        percent = 5.0 + 85.0 * min(done, total) / total
    elif state == "running":
        percent = 3.0
    else:
        percent = 0.0
    return Task(
        step=step,
        benchmark="GenEval Position-72",
        seed=None,
        state=state,
        phase=phase,
        percent=percent,
        detail=f"images={done}/{total}",
        eta_seconds=rate_eta(data, percent),
    )


def self_cycle_task(run_root: Path, step: int, seed: int) -> Task:
    variant = f"mask_flow_retention_{step}_prompt_conditioned"
    run_dir = run_root / f"step-{step}" / "self_cycle" / f"seed-{seed}" / variant
    data = load_json(run_dir / "status.json")
    state = state_for(run_dir, data)
    phase = str(data.get("phase", "not_started"))
    percent = 100.0 if state == "completed" else float(data.get("percent", 0.0))
    detail = (
        f"images={int(data.get('completed_images', 0))}/"
        f"{int(data.get('total_images', 689))} "
        f"questions={int(data.get('completed_questions', 0))}/"
        f"{int(data.get('total_questions', 1087))}"
    )
    return Task(
        step=step,
        benchmark="Self-cycle",
        seed=seed,
        state=state,
        phase=phase,
        percent=percent,
        detail=detail,
        eta_seconds=rate_eta(data, percent),
    )


def build_tasks(run_root: Path) -> list[Task]:
    return [
        task
        for step in STEPS
        for task in (
            geneval_task(run_root, step),
            *(self_cycle_task(run_root, step, seed) for seed in SEEDS),
        )
    ]


def queue_eta(tasks: list[Task], current_index: int | None) -> float:
    total = 0.0
    for index, task in enumerate(tasks):
        if task.state == "completed":
            continue
        if index == current_index and task.eta_seconds is not None:
            total += task.eta_seconds
        else:
            kind = "geneval" if task.seed is None else "self_cycle"
            total += HISTORICAL_SECONDS[kind] * (100.0 - task.percent) / 100.0
    return total


def render(run_root: Path, session: str) -> tuple[str, str]:
    tasks = build_tasks(run_root)
    completed = sum(task.state == "completed" for task in tasks)
    failed = sum(task.state == "failed" for task in tasks)
    current_index = next(
        (index for index, task in enumerate(tasks) if task.state == "running"), None
    )
    overall = sum(task.percent for task in tasks) / len(tasks)
    if failed or (run_root / "SWEEP_FAILED").is_file():
        queue_state = "failed"
    elif completed == len(tasks) or (run_root / "SWEEP_COMPLETED").is_file():
        queue_state = "completed"
    elif current_index is not None:
        queue_state = "running"
    else:
        queue_state = "pending"

    lines = [
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} ",
        f"queue={run_root.name} status={queue_state}",
        "model=Mask-logits + L_flow + teacher retention",
        f"total {bar(overall)} {overall:6.2f}% completed={completed}/{len(tasks)} "
        f"failed={failed} eta~={duration(queue_eta(tasks, current_index))}",
    ]
    if current_index is None:
        lines.append("current=none")
    else:
        current = tasks[current_index]
        seed = "n/a" if current.seed is None else str(current.seed)
        lines.extend(
            [
                f"current={current_index + 1}/{len(tasks)} step={current.step} "
                f"benchmark={current.benchmark} seed={seed} phase={current.phase}",
                f"current run {bar(current.percent)} {current.percent:6.2f}% "
                f"eta={duration(current.eta_seconds)} {current.detail}",
            ]
        )

    tags = {"completed": "DONE", "running": "RUN ", "failed": "FAIL", "pending": "WAIT"}
    lines.extend(["", "benchmarks:"])
    for index, task in enumerate(tasks, 1):
        seed = "n/a" if task.seed is None else str(task.seed)
        lines.append(
            f"[{tags[task.state]}] {index:>2}/{len(tasks)} step={task.step:<4} "
            f"{task.benchmark:<19} seed={seed:<8} "
            f"progress={task.percent:6.2f}% phase={task.phase}"
        )

    pane = subprocess.run(
        [
            "tmux", "list-panes", "-t", f"={session}", "-F",
            "#{pane_pid} #{pane_current_command} #{pane_dead}",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    lines.extend(["", f"tmux={pane or 'not found'}", f"gpu={gpu or 'unavailable'}"])
    return "\n".join(lines), queue_state


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("interval must be positive")
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    while True:
        output, state = render(run_root, args.session)
        if args.watch and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print(output, flush=True)
        if not args.watch or state in {"completed", "failed"}:
            return 1 if state == "failed" else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
