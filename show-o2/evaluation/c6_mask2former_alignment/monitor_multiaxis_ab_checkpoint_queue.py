#!/usr/bin/env python3
"""Show weighted progress and ETA for the multi-axis checkpoint A/B queue."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


WEIGHTS = {
    "preflight": 60.0,
    "eval": 950.0,
    "vqa": 650.0,
    "vsr": 40.0,
    "aggregate": 60.0,
}


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def bar(percent: float, width: int = 28) -> str:
    filled = min(width, max(0, round(width * percent / 100.0)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    value = max(0, round(seconds))
    days, remainder = divmod(value, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    prefix = f"{days}d " if days else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{seconds:02d}"


def fraction(data: dict[str, Any], completed: str, total: str) -> float:
    if data.get("status") == "completed":
        return 1.0
    denominator = float(data.get(total, 0) or 0)
    numerator = float(data.get(completed, 0) or 0)
    return min(1.0, max(0.0, numerator / denominator)) if denominator else 0.0


def task_kind(task: str) -> str:
    if task.startswith("eval__"):
        return "eval"
    if task.startswith("vqa__"):
        return "vqa"
    if task.startswith("vsr__") or task == "vsr_baseline_dev":
        return "vsr"
    return task


def task_weight(task: str) -> float:
    return WEIGHTS.get(task_kind(task), 60.0)


def self_cycle_progress(data: dict[str, Any]) -> tuple[float, float | None, str]:
    if data.get("status") == "completed" or data.get("phase") == "completed":
        return 1.0, 0.0, "completed"
    images = fraction(data, "completed_images", "total_images")
    questions = fraction(data, "completed_questions", "total_questions")
    phase = str(data.get("phase", data.get("status", "waiting")))
    if phase == "generation":
        sampler_step = float(data.get("sampler_step", 0) or 0)
        sampler_steps = float(data.get("sampler_steps", 0) or 0)
        active = float(data.get("active_generation_images", 0) or 0)
        total_images = float(data.get("total_images", 96) or 96)
        partial = active * sampler_step / sampler_steps if sampler_steps else 0.0
        images = min(1.0, images + partial / total_images)
        progress = 0.90 * images
    elif phase in {"qa", "scoring"}:
        progress = 0.90 + 0.10 * questions
    else:
        progress = max(0.90 * images, questions)
    eta = data.get("eta_seconds")
    return progress, float(eta) if eta is not None else None, phase


def active_progress(task: str | None, paths: list[str]) -> tuple[float, float | None, str]:
    if not task:
        return 0.0, None, "waiting"
    data = [load_json(Path(path)) for path in paths]
    if task_kind(task) == "eval":
        geneval = data[0] if data else {}
        self_cycle = data[1] if len(data) > 1 else {}
        geneval_fraction = fraction(geneval, "completed_images", "total_images")
        geneval_phase = str(geneval.get("phase", geneval.get("status", "waiting")))
        if geneval_phase.startswith("scoring"):
            geneval_fraction = max(geneval_fraction, 0.95)
        self_fraction, self_eta, self_phase = self_cycle_progress(self_cycle)
        progress = (420.0 * geneval_fraction + 530.0 * self_fraction) / 950.0
        eta_values: list[float] = []
        if geneval_fraction < 1.0:
            eta_values.append(
                float(geneval["eta_seconds"])
                if geneval.get("eta_seconds") is not None
                else 420.0 * (1.0 - geneval_fraction)
            )
        if self_cycle:
            eta_values.append(
                self_eta if self_eta is not None else 530.0 * (1.0 - self_fraction)
            )
        else:
            eta_values.append(530.0)
        phase = f"geneval:{geneval_phase} self-cycle:{self_phase}"
        return progress, sum(eta_values) if eta_values else None, phase
    current = data[0] if data else {}
    progress = float(current.get("percent", 0.0) or 0.0) / 100.0
    if current.get("status") == "completed":
        progress = 1.0
    eta = current.get("eta_seconds")
    phase = str(current.get("phase", current.get("status", "starting")))
    return max(0.0, min(1.0, progress)), float(eta) if eta is not None else None, phase


def training_progress(root: Path) -> tuple[float, float | None, list[str]]:
    config = load_json(root / "queue_config.json")
    configured_total = max(1, int(config.get("max_steps", 17100)))
    rows: list[str] = []
    fractions: list[float] = []
    remaining: list[float] = []
    reference_full: float | None = None
    statuses = []
    for route in ("g_adapter", "lora"):
        status = load_json(root / route / "status.json")
        state = str(status.get("status", "waiting"))
        total = max(1, int(status.get("total_steps", configured_total)))
        step = min(total, max(0, int(status.get("step", 0))))
        if state == "completed":
            step = total
        fraction_value = step / total
        fractions.append(fraction_value)
        if step > 0 and status.get("elapsed_seconds"):
            reference_full = float(status["elapsed_seconds"]) / step * total
        statuses.append((route, state, step, total, status))
    for route, state, step, total, status in statuses:
        eta = status.get("eta_seconds")
        if state == "completed":
            remaining.append(0.0)
        elif eta is not None:
            remaining.append(float(eta))
        elif reference_full is not None:
            remaining.append(reference_full)
        rows.append(
            f"  {route:<10} status={state:<10} step={step:5d}/{total:<5d} "
            f"progress={100.0 * step / total:6.2f}% eta={duration(float(eta) if eta is not None else None)}"
        )
    eta_total = sum(remaining) if len(remaining) == len(statuses) else None
    return sum(fractions) / len(fractions), eta_total, rows


def render(run_root: Path, session: str) -> str:
    status = load_json(run_root / "queue_status.json")
    if not status:
        return f"run={run_root}\nstatus=waiting_for_queue_status"
    planned = [str(value) for value in status.get("planned_task_ids", [])]
    completed = {str(value) for value in status.get("completed_task_ids", [])}
    current = status.get("current_task")
    active_fraction, active_eta, active_phase = active_progress(
        str(current) if current else None, status.get("active_status_paths", [])
    )
    total_work = sum(task_weight(task) for task in planned)
    completed_work = sum(task_weight(task) for task in planned if task in completed)
    if current in planned and current not in completed:
        completed_work += task_weight(str(current)) * active_fraction
    work_percent = 100.0 * completed_work / total_work if total_work else 0.0
    remaining = 0.0
    for task in planned:
        if task in completed:
            continue
        if task == current:
            historical = task_weight(task) * (1.0 - active_fraction)
            remaining += active_eta if active_eta is not None else historical
        else:
            remaining += task_weight(task)
    count_percent = 100.0 * (len(completed) + active_fraction) / len(planned) if planned else 0.0
    lines = [
        datetime.now().astimezone().isoformat(timespec="seconds"),
        f"run={run_root}",
        f"queue_status={status.get('status')} phase={status.get('phase')} current={current or 'none'}",
    ]
    if status.get("phase") == "waiting_for_training":
        train_root = Path(str(status.get("training_wait_root")))
        if status.get("wait_marker") == "BENCHMARK_QUEUE_COMPLETED":
            upstream = load_json(train_root / "queue_status.json")
            completed_upstream = int(upstream.get("completed_tasks", 0) or 0)
            total_upstream = int(upstream.get("total_tasks", 0) or 0)
            upstream_fraction = (
                completed_upstream / total_upstream if total_upstream else 0.0
            )
            lines.append(
                f"upstream benchmark {bar(100.0 * upstream_fraction)} "
                f"{100.0 * upstream_fraction:7.3f}% "
                f"completed={completed_upstream}/{total_upstream} "
                f"waiting for {status.get('wait_marker')}"
            )
            lines.append(
                f"detail={status.get('detail', status.get('wait_label', 'upstream benchmark'))}"
            )
        else:
            train_fraction, train_eta, train_rows = training_progress(train_root)
            lines.extend(
                [
                    f"upstream training {bar(100.0 * train_fraction)} {100.0 * train_fraction:7.3f}% eta~={duration(train_eta)}",
                    *train_rows,
                    f"benchmarks {bar(0.0)}   0.000% waiting for upstream QUEUE_COMPLETED",
                ]
            )
    else:
        finish = datetime.now().astimezone() + timedelta(seconds=remaining)
        lines.extend(
            [
                f"benchmark work  {bar(work_percent)} {work_percent:7.3f}% eta~={duration(remaining)} finish~={finish.isoformat(timespec='minutes')}",
                f"benchmark tasks {bar(count_percent)} {count_percent:7.3f}% completed={len(completed)}/{len(planned)}",
                f"current task    {bar(100.0 * active_fraction)} {100.0 * active_fraction:7.3f}% eta={duration(active_eta)} phase={active_phase}",
            ]
        )
    protocol = status.get("protocol", {})
    lines.append(f"steps={protocol.get('benchmark_steps')} routes={protocol.get('routes')} self-cycle=compact96@seed20260810")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    pane = subprocess.run(
        ["tmux", "list-panes", "-t", f"={session}", "-F", "#{pane_pid} #{pane_current_command} #{pane_dead}"],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    lines.extend([f"gpu={gpu or 'unavailable'}", f"tmux={pane or 'not_running'}"])
    if status.get("error"):
        lines.append(f"error={status['error']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--session", default="showo2_multiaxis_ab_benchmarks")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        print(render(args.run_root, args.session), flush=True)
        if not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
