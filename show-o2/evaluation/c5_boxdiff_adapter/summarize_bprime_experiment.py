#!/usr/bin/env python3
"""Combine B-prime training, VSR, Position-72, and old-B400 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--old-position-root", type=Path, required=True)
    parser.add_argument("--b1-position-root", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def detection_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    correct = 0
    both_present = 0
    by_relation: dict[str, dict[str, int]] = {}
    for row in rows:
        metadata = json.loads(row["metadata"])
        details = json.loads(row["details"])
        classes = [str(item["class"]) for item in metadata.get("include", ())]
        present = all(bool(details.get(name)) for name in classes)
        relation = "unknown"
        for item in metadata.get("include", ()):
            if item.get("position"):
                relation = str(item["position"][0])
                break
        bucket = by_relation.setdefault(relation, {"correct": 0, "present": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += int(bool(row["correct"]))
        bucket["present"] += int(present)
        correct += int(bool(row["correct"]))
        both_present += int(present)
    total = len(rows)
    return {
        "correct": correct,
        "both_objects_detected": both_present,
        "total": total,
        "accuracy": correct / total,
        "object_presence_rate": both_present / total,
        "by_relation": by_relation,
    }


def mean_window(records, key, start, stop):
    values = records[start:stop]
    return sum(float(row[key]) for row in values) / len(values)


def mean_raw_window(records, key, start, stop):
    values = records[start:stop]
    return sum(float(row["raw_losses"][key]) for row in values) / len(values)


def main() -> int:
    args = parse_args()
    selected = load(args.benchmark_root / "selected_checkpoint.json")
    vsr = load(args.benchmark_root / "vsr/summary.json")["variants"]
    quick = load(args.benchmark_root / "quick_position/summary.json")
    full = load(args.benchmark_root / "full_position/summary.json")
    audit = load(args.data_audit)
    b2_status = load(args.train_root / "b2/status.json")
    b2_records = load(args.train_root / "b2/bprime_position_metrics.json")["records"]
    old_summary = load(args.old_position_root / "summary.json")
    b1_summary = load(args.b1_position_root / "summary.json")

    result_paths = {
        "baseline": args.benchmark_root / "full_position/baseline/results/results.jsonl",
        "bprime": args.benchmark_root / "full_position/bprime_selected/results/results.jsonl",
        "old_b400": args.old_position_root / "b_400/results/results.jsonl",
        "b1_200": args.b1_position_root / "b1_200/results/results.jsonl",
    }
    position = {name: detection_summary(path) for name, path in result_paths.items()}
    transitions = full["paired_transitions_vs_baseline"]["bprime_selected"]
    payload = {
        "status": "completed",
        "protocol": "bprime-split-flow-position-v1",
        "selected": selected,
        "data": {
            "train_rows": audit["train_rows"],
            "validation_rows": audit["validation_rows"],
            "train_unique_images": audit["train_unique_images"],
            "image_id_overlap": audit["image_id_overlap"],
            "train_direction_counts_before_balance": audit["train_direction_counts_before_balance"],
        },
        "training": {
            "b1_steps": 200,
            "b2_steps": len(b2_records),
            "loss_weights": b2_status["loss_weights"],
            "calibration_grad_norms": b2_status["calibration_grad_norms"],
            "first50": {
                name: mean_raw_window(b2_records, name, 0, 50)
                for name in b2_records[0]["raw_losses"]
            },
            "last50": {
                name: mean_raw_window(b2_records, name, -50, len(b2_records))
                for name in b2_records[0]["raw_losses"]
            },
            "last_flow_drift_readonly": next(
                row["velocity_drift_readonly"]
                for row in reversed(b2_records)
                if row["velocity_drift_readonly"] is not None
            ),
        },
        "vsr_256": vsr,
        "position8": {
            "variants": quick["variants"],
            "paired_transitions_vs_baseline": quick["paired_transitions_vs_baseline"],
        },
        "position72": position,
        "bprime_paired_transitions_vs_baseline": transitions,
        "old_b400_summary": old_summary["variants"]["b_400"],
        "b1_control_summary": b1_summary["variants"]["b1_200"],
        "b1_control_transitions_vs_baseline": b1_summary["paired_transitions_vs_baseline"]["b1_200"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    baseline, b1, old, new = position["baseline"], position["b1_200"], position["old_b400"], position["bprime"]
    lines = [
        "# B-prime 分离 Flow/Position 实验结果",
        "",
        "## 结论表",
        "",
        "| 方案 | VSR-256 | Position-72 | 两目标均可检测 | 相对 baseline gain/loss/net |",
        "|---|---:|---:|---:|---:|",
        f"| baseline | {vsr['baseline']['correct']}/256 | {baseline['correct']}/288 | {baseline['both_objects_detected']}/288 | - |",
        f"| 旧 B400 joint flow+position | {vsr['old_b400']['correct']}/256 | {old['correct']}/288 | {old['both_objects_detected']}/288 | 旧记录见原 summary |",
        f"| B1 native anchor | {vsr['b1_200']['correct']}/256 | {b1['correct']}/288 | {b1['both_objects_detected']}/288 | {b1_summary['paired_transitions_vs_baseline']['b1_200']['gain']}/{b1_summary['paired_transitions_vs_baseline']['b1_200']['loss']}/{b1_summary['paired_transitions_vs_baseline']['b1_200']['net_gain']} |",
        f"| B-prime {selected['selected_variant']} | {vsr[selected['selected_variant']]['correct']}/256 | {new['correct']}/288 | {new['both_objects_detected']}/288 | {transitions['gain']}/{transitions['loss']}/{transitions['net_gain']} |",
        "",
        "## 训练事实",
        "",
        f"- 清洗训练集 {audit['train_rows']} 条、验证集 {audit['validation_rows']} 条，train/validation image overlap={audit['image_id_overlap']}。",
        f"- B2 first50 -> last50: box {payload['training']['first50']['box_g']:.6f} -> {payload['training']['last50']['box_g']:.6f}; relation {payload['training']['first50']['relation_g']:.6f} -> {payload['training']['last50']['relation_g']:.6f}; separation {payload['training']['first50']['separation']:.6f} -> {payload['training']['last50']['separation']:.6f}。",
        f"- 最后一个只读 velocity drift MSE={payload['training']['last_flow_drift_readonly']:.6f}；该项没有参与 backward。",
        "",
        "## 解释边界",
        "",
        "实际 B1→B2 的 Position 从 58/288 降到 53/288，两目标存在从 179/288 降到 178/288；同时 box/relation proxy 下降。因此本次 position-only G-hidden proxy 通过了优化检查但未通过外部生成 gate，B2 checkpoint 必须拒绝。配对差异未达到统计显著，结论是没有观察到增益，而不是证明该方向在所有设置下必然有害。",
    ]
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "selected": selected["selected_variant"],
        "vsr": vsr[selected["selected_variant"]],
        "position": new,
        "old_b400": old,
        "output": str(args.output_json),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
