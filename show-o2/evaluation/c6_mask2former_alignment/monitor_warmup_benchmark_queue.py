#!/usr/bin/env python3
"""Show total and active-task progress for the warmup benchmark queue."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path


# Seconds per scheduler task, calibrated from completed runs on this machine.
# The queue mixes cheap bookkeeping with long generation jobs, so task-count
# progress is not a useful proxy for wall-clock progress.
HISTORICAL_SECONDS = {
    "prepare_coco500": 5.0,
    "replay_step5000": 180.0,
    "vsr_dev": 35.0,
    "vsr_test": 70.0,
    "geneval": 412.0,       # Position-72: 288 images at 12 sampling steps.
    "self_cycle": 3745.0,  # 689 images plus 1,087 frozen-U questions.
    "visor": 420.0,        # 200 images at 12 steps plus OWL-ViT scoring.
    "vqascore": 1550.0,    # Baseline plus eleven 288-image checkpoints.
    "coco500": 2680.0,     # 500 images at 50 steps; measured at ~5.3 s/image.
    "fid_kid": 900.0,
    "cmmd": 1800.0,
    "aggregate": 10.0,
}
SELF_CYCLE_GENERATION_SECONDS = 3590.0
SELF_CYCLE_QA_SECONDS = HISTORICAL_SECONDS["self_cycle"] - SELF_CYCLE_GENERATION_SECONDS


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def bar(percent: float, width: int = 28) -> str:
    filled = max(0, min(width, round(width * percent / 100)))
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


def task_seconds(task_id: str) -> float:
    direct = {
        "prepare_coco500": "prepare_coco500",
        "replay_step5000": "replay_step5000",
        "vqascore_all": "vqascore",
        "fid_kid_all": "fid_kid",
        "cmmd_all": "cmmd",
        "aggregate": "aggregate",
    }
    if task_id in direct:
        return HISTORICAL_SECONDS[direct[task_id]]
    if task_id.startswith("vsr_dev_") or task_id == "vsr_baseline_dev":
        return HISTORICAL_SECONDS["vsr_dev"]
    if task_id.startswith("vsr_test_") or task_id == "vsr_baseline_test":
        return HISTORICAL_SECONDS["vsr_test"]
    if task_id.startswith("geneval_"):
        return HISTORICAL_SECONDS["geneval"]
    if task_id.startswith("self_cycle_"):
        return HISTORICAL_SECONDS["self_cycle"]
    if task_id.startswith("visor_"):
        return HISTORICAL_SECONDS["visor"]
    if task_id.startswith("coco500_"):
        return HISTORICAL_SECONDS["coco500"]
    return 60.0


def status_percent(data: dict) -> float:
    percent = data.get("percent")
    if percent is None:
        completed = data.get("completed_images", data.get("completed", 0))
        total = data.get("total_images", data.get("total", 0))
        percent = 100.0 * float(completed) / float(total) if total else 0.0
    return max(0.0, min(100.0, float(percent)))


def self_cycle_progress(data: dict) -> tuple[float, str, float]:
    total_images = max(1, int(data.get("total_images", 689)))
    total_questions = max(1, int(data.get("total_questions", 1087)))
    completed_images = max(0, min(total_images, int(data.get("completed_images", 0))))
    completed_questions = max(
        0, min(total_questions, int(data.get("completed_questions", 0)))
    )
    active_images = max(0, int(data.get("active_generation_images", 0)))
    sampler_step = max(0, int(data.get("sampler_step", 0)))
    sampler_steps = max(0, int(data.get("sampler_steps", 0)))
    partial_images = (
        active_images * min(sampler_step, sampler_steps) / sampler_steps
        if sampler_steps else 0.0
    )
    generation_fraction = min(
        1.0, (completed_images + partial_images) / total_images
    )
    qa_fraction = completed_questions / total_questions
    remaining = (
        SELF_CYCLE_GENERATION_SECONDS * (1.0 - generation_fraction)
        + SELF_CYCLE_QA_SECONDS * (1.0 - qa_fraction)
    )
    percent = 100.0 * (1.0 - remaining / HISTORICAL_SECONDS["self_cycle"])
    detail = (
        f"self_cycle:{data.get('phase', data.get('status', 'unknown'))} "
        f"images={completed_images}/{total_images} "
        f"questions={completed_questions}/{total_questions}"
    )
    return max(0.0, min(100.0, percent)), detail, max(0.0, remaining)


def active_progress(task_id: str | None, paths: list[str]) -> tuple[float, str, float | None]:
    candidates = []
    for value in paths:
        data = load(Path(value))
        if not data:
            continue
        candidates.append((
            status_percent(data),
            f"{Path(value).parent.name}:{data.get('phase', data.get('status', 'unknown'))}",
            float(data["eta_seconds"]) if data.get("eta_seconds") is not None else None,
        ))
    if not candidates:
        return 0.0, "starting", None

    # The runner's generic work-unit ETA treats one 50-step image as equal to
    # one cheap U-head question and includes time spent paused between queue
    # restarts. Use measured phase costs instead: generation dominates (~96%).
    if task_id and task_id.startswith("self_cycle_"):
        data = load(Path(paths[0]))
        if data:
            return self_cycle_progress(data)

    # VISOR exposes generation and OWL-ViT as separate status files. Generation
    # dominates runtime, so combine both phases into one scheduler-task percent.
    if task_id and task_id.startswith("visor_") and len(candidates) >= 2:
        generation, scoring = candidates[0], candidates[1]
        percent = 0.95 * generation[0] + 0.05 * scoring[0]
        active = generation if generation[0] < 100.0 else scoring
        eta = active[2]
        if generation[0] < 100.0:
            generation_eta = eta or HISTORICAL_SECONDS["visor"] * (100.0 - generation[0]) / 100.0
            eta = generation_eta + 225.0
        return percent, active[1], eta

    unfinished = [item for item in candidates if item[0] < 100.0]
    return unfinished[0] if unfinished else candidates[-1]


def queue_progress(data: dict, active_percent: float,
                   active_eta: float | None) -> tuple[float, float]:
    planned = [str(value) for value in data.get("planned_task_ids", [])]
    completed = {str(value) for value in data.get("completed_task_ids", [])}
    current = data.get("current_task")
    total_work = sum(task_seconds(task_id) for task_id in planned)
    completed_work = sum(task_seconds(task_id) for task_id in planned if task_id in completed)
    if current in planned and current not in completed:
        completed_work += task_seconds(str(current)) * active_percent / 100.0

    eta = 0.0
    for task_id in planned:
        if task_id in completed:
            continue
        if task_id == current:
            historical_remaining = task_seconds(task_id) * (100.0 - active_percent) / 100.0
            eta += active_eta if active_eta is not None else historical_remaining
        else:
            eta += task_seconds(task_id)
    percent = 100.0 * completed_work / total_work if total_work else 0.0
    return percent, eta


def render(root: Path, session: str) -> str:
    data = load(root / "queue_status.json")
    if not data:
        return f"run={root}\nstatus=waiting_for_queue_status"
    completed, total = int(data.get("completed_tasks", 0)), int(data.get("total_tasks", 0))
    current = data.get("current_task")
    partial, detail, current_eta = active_progress(current, data.get("active_status_paths", []))
    if current_eta is None and current in data.get("planned_task_ids", []):
        current_eta = task_seconds(str(current)) * (100.0 - partial) / 100.0
    percent, eta = queue_progress(data, partial, current_eta)
    count_percent = 100.0 * (completed + partial / 100.0) / total if total else 0.0
    finish = datetime.now().astimezone() + timedelta(seconds=eta)
    protocol = data.get("protocol", {})
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    ).stdout.strip()
    pane = subprocess.run(
        ["tmux", "list-panes", "-t", f"={session}", "-F", "#{pane_pid} #{pane_current_command} #{pane_dead}"],
        text=True, capture_output=True, check=False,
    ).stdout.strip()
    return "\n".join([
        time.strftime("%Y-%m-%dT%H:%M:%S%z"), f"run={root}",
        f"status={data.get('status')} task={current} detail={detail}",
        f"work  {bar(percent)} {percent:7.3f}% eta~={duration(eta)} finish~={finish.isoformat(timespec='minutes')}",
        f"tasks {bar(count_percent)} {count_percent:7.3f}% completed={completed}/{total}",
        f"current {bar(partial)} {partial:7.3f}% eta={duration(current_eta)}",
        f"protocol=steps:{protocol.get('steps')} excluded:{protocol.get('excluded_historical_steps')} self-cycle-seeds:{protocol.get('self_cycle_seeds')}",
        f"VISOR:{protocol.get('visor_images_per_variant')}@{protocol.get('visor_inference_steps')}step({protocol.get('visor_scope')}) COCO:{protocol.get('coco_images_per_checkpoint')}({protocol.get('coco_scope')})",
        f"gpu={gpu or 'unavailable'}", f"tmux={pane or 'not_running'}",
        f"error={data['error']}" if data.get("error") else "",
    ]).rstrip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--session", default="showo2_warmup_benchmarks")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
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
