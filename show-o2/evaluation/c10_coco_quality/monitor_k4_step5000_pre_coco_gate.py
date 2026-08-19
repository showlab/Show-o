#!/usr/bin/env python3
"""Monitor the Mask+Flow-500 -> K4-5000 benchmark -> COCO queue gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SEEDS = (20260810, 20260811, 20260812, 20260813)
VARIANT = "mask_flow_k4_5000_prompt_conditioned"
HISTORICAL_SECONDS = {"geneval": 420.0, "self_cycle": 3750.0}


@dataclass(frozen=True)
class Task:
    label: str
    state: str
    phase: str
    percent: float
    detail: str
    eta: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--quality-run", type=Path, required=True)
    parser.add_argument("--session", default="showo2_k4_5000_pre_coco_gate")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def load_env(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    return dict(line.split("=", 1) for line in lines if "=" in line)


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


def parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def rate_eta(data: dict, percent: float) -> float | None:
    if data.get("eta_seconds") is not None:
        return float(data["eta_seconds"])
    started = parse_time(data.get("started_at"))
    if started is None or not 0.0 < percent < 100.0:
        return None
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return elapsed * (100.0 - percent) / percent


def quality_task(quality_run: Path) -> Task:
    data = load_json(quality_run / "generation" / "mask_flow_500" / "status.json")
    done = int(data.get("completed_images", 0))
    total = int(data.get("total_images", 5000)) or 5000
    state = "completed" if data.get("status") == "completed" else (
        "failed" if data.get("status") == "failed" else ("running" if data else "pending")
    )
    percent = 100.0 if state == "completed" else 100.0 * min(done, total) / total
    return Task(
        "Mask+Flow step-500 COCO generation",
        state,
        str(data.get("phase", "not_started")),
        percent,
        f"images={done}/{total} sampler={data.get('sampler_step', 0)}/{data.get('sampler_steps', 49)}",
        rate_eta(data, percent),
    )


def geneval_task(run_root: Path) -> Task:
    run_dir = run_root / "geneval_position72"
    data = load_json(run_dir / "status.json")
    completed = (run_dir / "BENCHMARK_COMPLETED").is_file() or data.get("status") == "completed"
    state = "completed" if completed else ("failed" if data.get("status") == "failed" else ("running" if data else "pending"))
    phase = str(data.get("phase", "not_started"))
    done = int(data.get("completed_images", 0))
    total = int(data.get("total_images", 288)) or 288
    if completed:
        percent = 100.0
    elif phase.startswith("scoring"):
        percent = 95.0
    elif phase == "generation":
        percent = 5.0 + 85.0 * min(done, total) / total
    elif state == "running":
        percent = 3.0
    else:
        percent = 0.0
    return Task("K4 step-5000 GenEval Position-72", state, phase, percent, f"images={done}/{total}", rate_eta(data, percent))


def self_cycle_task(run_root: Path, seed: int) -> Task:
    run_dir = run_root / "self_cycle" / f"seed-{seed}" / VARIANT
    data = load_json(run_dir / "status.json")
    completed = (run_dir / "BENCHMARK_COMPLETED").is_file() or data.get("status") == "completed"
    state = "completed" if completed else ("failed" if data.get("status") == "failed" else ("running" if data else "pending"))
    percent = 100.0 if completed else float(data.get("percent", 0.0))
    detail = (
        f"images={int(data.get('completed_images', 0))}/{int(data.get('total_images', 689))} "
        f"questions={int(data.get('completed_questions', 0))}/{int(data.get('total_questions', 1087))}"
    )
    return Task(f"K4 step-5000 Self-cycle seed={seed}", state, str(data.get("phase", "not_started")), percent, detail, rate_eta(data, percent))


def render(run_root: Path, quality_run: Path, session: str) -> tuple[str, str]:
    gate = load_env(run_root / "gate.status")
    quality = quality_task(quality_run)
    benchmarks = [geneval_task(run_root), *(self_cycle_task(run_root, seed) for seed in SEEDS)]
    completed = sum(task.state == "completed" for task in benchmarks)
    failed = sum(task.state == "failed" for task in benchmarks)
    overall = sum(task.percent for task in benchmarks) / len(benchmarks)
    current = next((task for task in benchmarks if task.state == "running"), None)
    benchmark_eta = 0.0
    for task in benchmarks:
        if task.state == "completed":
            continue
        if task is current and task.eta is not None:
            benchmark_eta += task.eta
        elif "GenEval" in task.label:
            benchmark_eta += HISTORICAL_SECONDS["geneval"] * (100.0 - task.percent) / 100.0
        else:
            benchmark_eta += HISTORICAL_SECONDS["self_cycle"] * (100.0 - task.percent) / 100.0
    total_eta = benchmark_eta + (quality.eta or 0.0 if quality.state != "completed" else 0.0)
    state = gate.get("status", "initializing")
    if (run_root / "PRE_COCO_GATE_COMPLETED").is_file():
        state = "completed"
    elif (run_root / "PRE_COCO_GATE_FAILED").is_file():
        state = "failed"

    lines = [
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        f"gate={run_root.name} status={state} phase={gate.get('phase', 'initializing')}",
        f"precondition {bar(quality.percent)} {quality.percent:6.2f}% {quality.detail} eta={duration(quality.eta)}",
        f"K4-5000 benchmarks {bar(overall)} {overall:6.2f}% completed={completed}/5 failed={failed} eta~={duration(total_eta)}",
    ]
    if current is not None:
        lines.append(f"current={current.label} phase={current.phase} progress={current.percent:.2f}% eta={duration(current.eta)} {current.detail}")
    elif quality.state != "completed":
        lines.append("current=waiting for Mask+Flow step-500 COCO generation to complete")
    else:
        lines.append("current=validation/resume boundary")

    tags = {"completed": "DONE", "running": "RUN ", "failed": "FAIL", "pending": "WAIT"}
    lines.extend(["", "tasks:", f"[{tags[quality.state]}] 0/5 {quality.label:<48} progress={quality.percent:6.2f}% phase={quality.phase}"])
    for index, task in enumerate(benchmarks, 1):
        lines.append(f"[{tags[task.state]}] {index}/5 {task.label:<48} progress={task.percent:6.2f}% phase={task.phase}")

    pane = subprocess.run(
        ["tmux", "list-panes", "-t", f"={session}", "-F", "#{pane_pid} #{pane_current_command} #{pane_dead}"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    k4_generation = load_json(quality_run / "generation" / "mask_flow_k4_5000" / "status.json")
    lines.extend(
        [
            "",
            f"gate_tmux={pane or 'not found'}",
            f"K4_COCO_phase={k4_generation.get('phase', 'not_started')}",
            f"gpu={gpu or 'unavailable'}",
        ]
    )
    return "\n".join(lines), state


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("interval must be positive")
    while True:
        output, state = render(args.run_root.resolve(), args.quality_run.resolve(), args.session)
        if args.watch and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print(output, flush=True)
        if not args.watch or state in {"completed", "failed"}:
            return 1 if state == "failed" else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
