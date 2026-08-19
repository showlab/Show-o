#!/usr/bin/env python3
"""Monitor C6 quick gate and optional Position-72 benchmark."""

import argparse, json, subprocess, time
from pathlib import Path


def load(path):
    try: return json.loads(path.read_text())
    except Exception: return {}


def bar(done, total, width=40):
    filled = int(width * min(done, total) / total) if total else 0
    return "[" + "#" * filled + "." * (width - filled) + "]"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--session", default="showo2_mask2former_benchmark")
    args=parser.parse_args()
    state=(args.run_root/"benchmark.status").read_text().strip() if (args.run_root/"benchmark.status").exists() else "pending"
    quick=load(args.run_root/"quick_position/status.json")
    full=load(args.run_root/"full_position/status.json")
    gate=load(args.run_root/"quick_gate.json")
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z")); print()
    print(f"Run:       {args.run_root.name}")
    print(f"Queue:     {state}")
    qd,qt=int(quick.get("completed_images",0)),int(quick.get("total_images",192))
    fd,ft=int(full.get("completed_images",0)),int(full.get("total_images",864))
    print(f"Quick:     {qd}/{qt} {bar(qd,qt)} {100*qd/qt:6.2f}%")
    if full or gate.get("selected_variant"):
        print(f"Full:      {fd}/{ft} {bar(fd,ft)} {100*fd/ft:6.2f}%")
    if gate:
        print(f"Gate:      {gate.get('status')} selected={gate.get('selected_variant')}")
        for name, values in gate.get("variants",{}).items():
            print(f"  {name:12s} correct={values['correct']:2d}/{values['total']} both={values['both_objects_detected']:2d} gain/loss/net={values['gain_vs_control']}/{values['loss_vs_control']}/{values['net_gain_vs_control']}")
    print()
    subprocess.run(["tmux","list-panes","-t",f"={args.session}","-F","tmux: PID=#{pane_pid} cmd=#{pane_current_command} dead=#{pane_dead}"],check=False)
    subprocess.run(["nvidia-smi","--query-gpu=index,name,utilization.gpu,memory.used,memory.total","--format=csv,noheader"],check=False)


if __name__ == "__main__": main()
