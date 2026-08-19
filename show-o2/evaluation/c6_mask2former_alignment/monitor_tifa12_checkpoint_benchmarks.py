#!/usr/bin/env python3
"""Monitor the TIFA-12 training waiter and six-checkpoint benchmark queue."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import monitor_multiaxis_ab_checkpoint_queue as common


# The old VQAScore weight covered four variants per task; this queue scores two.
common.WEIGHTS["vqa"] = 350.0


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def pane(session: str) -> str:
    return subprocess.run(
        ["tmux", "list-panes", "-t", f"={session}", "-F", "#{pane_pid} #{pane_current_command} #{pane_dead}"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip() or "not_running"


def gpu() -> str:
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip() or "unavailable"


def training_lines(root: Path) -> list[str]:
    queue = load_json(root / "queue_status.json")
    tasks = queue.get("tasks", [])
    maximum = max(1, int(queue.get("max_steps_per_route", 34012)))
    completed_units = 0
    rows: list[str] = []
    current_eta: float | None = None
    current_rate: float | None = None
    waiting_routes = 0
    for index, task in enumerate(tasks, 1):
        label = str(task.get("label", "unknown"))
        status = load_json(root / label / "status.json")
        state = str(status.get("status", task.get("status", "waiting")))
        step = int(status.get("step", maximum if state == "completed" else 0))
        if state == "completed":
            step = maximum
        step = min(maximum, max(0, step))
        completed_units += step
        eta = status.get("eta_seconds")
        if state == "running":
            current_eta = float(eta) if eta is not None else None
            rate = status.get("throughput_steps_per_second")
            current_rate = float(rate) if rate else None
        elif state == "waiting":
            waiting_routes += 1
        marker = {"completed": "DONE", "running": "RUN ", "failed": "FAIL"}.get(state, "WAIT")
        rows.append(
            f"[{marker}] {index}/{len(tasks)} {label:22s} step={step:5d}/{maximum} "
            f"progress={100.0 * step / maximum:7.3f}% "
            f"eta={common.duration(float(eta) if eta is not None else None)}"
        )
    total_units = max(1, maximum * max(1, len(tasks)))
    overall = 100.0 * completed_units / total_units
    overall_eta = current_eta
    if overall_eta is not None and current_rate:
        overall_eta += waiting_routes * maximum / current_rate
    return [
        f"training_run={root}",
        f"training_status={queue.get('status', 'unknown')} current={queue.get('current', 'none')} "
        f"overall={overall:.3f}% eta~={common.duration(overall_eta)}",
        *rows,
    ]


def render(run_root: Path, benchmark_session: str, training_session: str) -> str:
    status = load_json(run_root / "queue_status.json")
    if not status:
        return f"run={run_root}\nstatus=waiting_for_queue_status"
    if status.get("phase") != "waiting_for_training":
        return common.render(run_root, benchmark_session)
    planned = status.get("planned_task_ids", [])
    training_root = Path(str(status.get("training_wait_root")))
    protocol = status.get("protocol", {})
    lines = [datetime.now().astimezone().isoformat(timespec="seconds")]
    lines.extend(training_lines(training_root))
    lines.extend(
        [
            f"benchmark_run={run_root}",
            f"benchmark_status={status.get('status')} phase={status.get('phase')}",
            f"benchmark tasks {common.bar(0.0)}   0.000% completed=0/{len(planned)} waiting for QUEUE_COMPLETED",
            f"steps={protocol.get('benchmark_steps')} routes={protocol.get('routes')} self-cycle=compact96@seed20260810",
            f"gpu={gpu()}",
            f"training_tmux={pane(training_session)}",
            f"benchmark_tmux={pane(benchmark_session)}",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--session", default="showo2_tifa12_posttrain_benchmarks")
    parser.add_argument("--training-session", default="showo2_tifa12_self_cycle")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        print(render(args.run_root, args.session, args.training_session), flush=True)
        status = load_json(args.run_root / "queue_status.json")
        if not args.watch or status.get("status") in {"completed", "failed"}:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
