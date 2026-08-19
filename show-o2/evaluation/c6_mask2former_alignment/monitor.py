#!/usr/bin/env python3
"""Display progress and rolling metrics for online Mask2Former alignment."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def bar(value: int, total: int, width: int = 40) -> str:
    filled = width if total <= 0 else int(width * min(value, total) / total)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def duration(seconds):
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_tail(path: Path, count: int = 50):
    try:
        lines = path.read_text().splitlines()[-count:]
        return [json.loads(line) for line in lines if line]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def mean(rows, getter):
    values = [getter(row) for row in rows]
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--session", default="showo2_mask2former_5k")
    args = parser.parse_args()
    status = load_json(args.run_dir / "status.json")
    rows = load_tail(args.run_dir / "metrics.jsonl")
    step, total = int(status.get("step", 0)), int(status.get("total_steps", 5000))
    percent = 100.0 * step / total if total else 0.0
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print()
    print(f"Run:       {args.run_dir.name}")
    print(f"Status:    {status.get('status', 'pending')}")
    print(f"Phase:     {status.get('phase', 'pending')}")
    print(f"Progress:  {step}/{total} {bar(step, total)} {percent:6.2f}%")
    print(f"Elapsed:   {duration(status.get('elapsed_seconds'))}")
    print(f"ETA:       {duration(status.get('eta_seconds'))}")
    if rows:
        latest = rows[-1]
        valid = sum(bool(row.get("teacher_valid")) for row in rows)
        object_valid = sum(
            bool(row.get("teacher_object_valid", row.get("teacher_valid")))
            for row in rows
        )
        relation_valid = sum(
            bool(row.get("teacher_relation_valid", row.get("teacher_valid")))
            for row in rows
        )
        print()
        print(f"Current:   {latest.get('relation')}  t={latest.get('timestep', 0):.3f}")
        print(f"Loss:      {latest.get('loss', 0):.6f}  rolling50={mean(rows, lambda r: r.get('loss')):.6f}")
        flow_value = latest.get("flow_mse")
        if flow_value is not None:
            flow_weighted = latest.get("weighted_losses", {}).get("flow", 0.0)
            flow_rolling = mean(rows, lambda row: row.get("flow_mse"))
            print(
                f"Flow:      raw={float(flow_value):.6f} weighted={float(flow_weighted):.6f} "
                f"rolling50={flow_rolling:.6f}"
            )
        retention_value = latest.get("teacher_retention_mse")
        if retention_value is not None:
            retention_weighted = latest.get("weighted_losses", {}).get("teacher_retention", 0.0)
            retention_rolling = mean(rows, lambda row: row.get("teacher_retention_mse"))
            print(
                f"Retention: raw={float(retention_value):.6e} weighted={float(retention_weighted):.6e} "
                f"rolling50={retention_rolling:.6e}"
            )
        print(
            f"Relation:  student={latest['details']['student_relation_score'][0]:.4f} "
            f"teacher={latest['details']['teacher_relation_score'][0]:.4f}"
        )
        print(
            f"Presence:  student={latest['details']['student_presence'][0]} "
            f"teacher={latest['details']['teacher_presence'][0]}"
        )
        detector_scores = latest["details"].get("student_best_detector_scores")
        if detector_scores:
            base_target = float(latest["details"].get("object_detection_target", 0.0))
            targets = latest["details"].get("object_detection_targets", [[base_target]])
            object_loss = float(latest.get("raw_losses", {}).get("object_detect", 0.0))
            rolling_object = mean(
                rows, lambda row: row.get("raw_losses", {}).get("object_detect")
            )
            print(
                f"Objects:   best={detector_scores[0]} targets={targets[0]} "
                f"base={base_target:.3f} loss={object_loss:.6f} "
                f"rolling50={rolling_object:.6f}"
            )
        print(f"Gradient:  {latest.get('grad_g_spatial', 0):.4e}  lr={latest.get('lr_g_spatial', 0):.3e}")
        print(
            f"Teacher:   object={object_valid}/{len(rows)} "
            f"relation={relation_valid}/{len(rows)} "
            f"joint={valid}/{len(rows)}"
        )
        print(f"GPU peak:  {latest.get('peak_cuda_memory_bytes', 0) / 2**30:.2f} GiB")
    print()
    subprocess.run(["tmux", "list-panes", "-t", f"={args.session}", "-F", "tmux: PID=#{pane_pid} cmd=#{pane_current_command} dead=#{pane_dead}"], check=False)
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], check=False)
    error = status.get("error")
    if error:
        print(f"\nError: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
