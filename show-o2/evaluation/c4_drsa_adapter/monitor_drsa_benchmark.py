#!/usr/bin/env python3
"""Monitor DRSA generation scoring and VSR checkpoint sweep progress."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


RUN_ROOT = Path("/root/autodl-tmp/runs/showo2_drsa_benchmark")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def bar(value: int, total: int, width: int = 40) -> str:
    ratio = min(max(value / total if total else 0.0, 0.0), 1.0)
    filled = round(ratio * width)
    return "[" + "#" * filled + "." * (width - filled) + f"] {ratio:6.2%}"


def duration(seconds) -> str:
    if seconds is None:
        return "pending"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=3
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def image_count(run_dir: Path, variant: str) -> int:
    return sum(1 for path in (run_dir / variant / "geneval").glob("*/samples/*.png") if path.is_file())


def scored_count(run_dir: Path, variant: str) -> tuple[int, int]:
    rows = []
    path = run_dir / variant / "results/results.jsonl"
    if path.is_file():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return sum(bool(row["correct"]) for row in rows), len(rows)


def generation_eta(status: dict, completed: int, total: int):
    if completed <= 0 or total <= completed or not status.get("started_at"):
        return 0 if total and completed >= total else None
    started = datetime.fromisoformat(status["started_at"]).timestamp()
    elapsed = max(0.0, time.time() - started)
    return elapsed / completed * (total - completed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "train"), nargs="?", default="train")
    parser.add_argument("run_dir", nargs="?", type=Path)
    args = parser.parse_args()
    marker = RUN_ROOT / f"latest_{args.mode}.txt"
    run_dir = args.run_dir or Path(marker.read_text().strip())
    generation_dir = run_dir / "generation"
    understanding_dir = run_dir / "understanding"
    generation_status = read_json(generation_dir / "status.json")
    protocol = read_json(generation_dir / "protocol.json")
    understanding_status = read_json(understanding_dir / "status.json")
    variants = tuple((protocol.get("variants") or {}).keys())
    total_images = int(generation_status.get("total_images", protocol.get("total_images", 0)))
    counts = {variant: image_count(generation_dir, variant) for variant in variants}
    baseline_reused = int(protocol.get("baseline_reused_images", 0))
    if run_dir.name.startswith("drsa-benchmark-20k-") and "baseline" in counts:
        baseline_reused = int(protocol.get("baseline_reused_images", counts["baseline"]))
    generated = sum(counts.values())
    generated_new = max(0, generated - baseline_reused)
    session = (
        "showo2_drsa_benchmark_20k"
        if run_dir.name.startswith("drsa-benchmark-20k-")
        else f"showo2_drsa_benchmark_{args.mode}"
    )
    tmux_info = command_output([
        "tmux", "list-panes", "-t", f"={session}",
        "-F", "pid=#{pane_pid} command=#{pane_current_command} dead=#{pane_dead}",
    ]) or "not running"
    gpu = command_output([
        "nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]) or "unavailable"

    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print()
    print(f"Run:       {run_dir.name}")
    print(f"tmux:      {tmux_info}")
    print(f"GPU:       {gpu}")
    print()
    print(f"Generation status: {generation_status.get('status', 'starting')}")
    print(f"Generation phase:  {generation_status.get('phase', 'initializing')}")
    print(f"Generated: {generated}/{total_images} (new {generated_new}) {bar(generated, total_images)}")
    print(f"Gen ETA:   {duration(generation_eta(generation_status, generated_new, total_images - baseline_reused))}")
    if generation_status.get("phase") == "generation":
        print(
            f"Current:   {generation_status.get('variant')} "
            f"source={generation_status.get('source_index')} "
            f"relation={generation_status.get('relation')}"
        )
        current_step = int(generation_status.get("euler_step", 0))
        total_steps = int(generation_status.get("euler_steps", 0))
        print(f"Euler:     {current_step}/{total_steps} {bar(current_step, total_steps)}")
    if variants:
        per_variant = int(protocol.get("prompts", 0)) * int(protocol.get("samples_per_prompt", 0))
        width = max(len(name) for name in variants)
        print()
        for variant in variants:
            correct, scored = scored_count(generation_dir, variant)
            score = f"{correct / scored:.2%}" if scored else "pending"
            print(f"{variant:<{width}} images={counts[variant]:4d}/{per_variant} score={correct}/{scored} ({score})")

    print()
    vsr_completed = int(understanding_status.get("completed_predictions", 0))
    vsr_total = int(understanding_status.get("total_predictions", 0))
    print(f"VSR status: {understanding_status.get('status', 'queued')}")
    print(f"VSR phase:  {understanding_status.get('phase', 'queued')}")
    print(f"VSR:        {vsr_completed}/{vsr_total} {bar(vsr_completed, vsr_total)}")
    print(f"VSR ETA:    {duration(understanding_status.get('eta_seconds'))}")
    if understanding_status.get("variant"):
        print(
            f"VSR current: {understanding_status['variant']} "
            f"{understanding_status.get('variant_completed', 0)}/"
            f"{understanding_status.get('variant_total', 0)}"
        )
    error = generation_status.get("error") or understanding_status.get("error")
    if error:
        print(f"\nError:     {error}")
    print(f"Log:       {run_dir / 'run.log'}")
    log_path = run_dir / "run.log"
    if log_path.is_file():
        lines = log_path.read_text(errors="replace").splitlines()[-8:]
        if lines:
            print("\nRecent log:")
            print("\n".join(lines))


if __name__ == "__main__":
    main()
