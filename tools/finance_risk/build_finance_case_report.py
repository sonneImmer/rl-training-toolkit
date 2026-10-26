#!/usr/bin/env python3
"""Build local RL case-study material from evaluations and rollout evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRIC_FIELDS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f2",
    "false_negative_rate",
    "format_valid_rate",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
]


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=/path/to/metrics.json")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=/path/to/metrics.json")
    return name, Path(raw_path)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def metric_row(name: str, path: Path) -> dict:
    metrics = load_json(path)
    row = {"name": name, "metrics_path": str(path.resolve())}
    row.update({field: metrics.get(field) for field in METRIC_FIELDS})
    return row


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_number(value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation", action="append", type=parse_named_path, required=True)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--rollout-summary", type=Path)
    parser.add_argument("--training-step", action="append", type=json.loads, default=[])
    parser.add_argument("--observation", action="append", default=[])
    parser.add_argument("--decision", action="append", default=[])
    parser.add_argument("--limitation", action="append", default=[])
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    metric_rows = [metric_row(name, path) for name, path in args.evaluation]
    by_name = {row["name"]: row for row in metric_rows}
    if args.selected not in by_name:
        raise ValueError(f"selected evaluation not found: {args.selected}")

    selected = by_name[args.selected]
    deltas = []
    for row in metric_rows:
        if row["name"] == args.selected:
            continue
        deltas.append({
            "selected": args.selected,
            "baseline": row["name"],
            **{
                f"delta_{field}": selected[field] - row[field]
                for field in ("accuracy", "precision", "recall", "f1", "f2", "false_negative_rate")
            },
            "delta_false_negative": selected["false_negative"] - row["false_negative"],
            "delta_false_positive": selected["false_positive"] - row["false_positive"],
        })

    rollout_steps = []
    if args.rollout_summary:
        rollout_steps = load_json(args.rollout_summary).get("steps", [])

    evidence = {
        "title": args.title,
        "selected": args.selected,
        "evaluations": metric_rows,
        "deltas": deltas,
        "rollout_steps": rollout_steps,
        "training_steps": args.training_step,
        "run_manifest": load_json(args.run_manifest) if args.run_manifest else None,
        "observations": args.observation,
        "decisions": args.decision,
        "limitations": args.limitation,
    }
    (args.output / "case_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output / "metrics_comparison.csv", metric_rows, ["name", *METRIC_FIELDS, "metrics_path"])
    if args.training_step:
        write_csv(
            args.output / "training_stability.csv",
            args.training_step,
            ["step", "reward_mean", "advantage_mean", "entropy", "kl_loss", "grad_norm", "clip_fraction", "response_clip_ratio"],
        )
    if rollout_steps:
        rollout_rows = []
        for step in rollout_steps:
            rollout_rows.append({
                "step": step.get("step"),
                "reward_mean": step.get("reward_mean"),
                "format_valid_rate": step.get("format_valid_rate"),
                "bankruptcy_outputs": step.get("prediction_counts", {}).get("BANKRUPTCY", 0),
                "healthy_outputs": step.get("prediction_counts", {}).get("HEALTHY", 0),
                "effective_group_rate": step.get("effective_group_rate"),
                "constant_group_rate": step.get("constant_group_rate"),
            })
        write_csv(args.output / "rollout_steps.csv", rollout_rows, list(rollout_rows[0]))

    lines = [f"# {args.title}", "", "## 结论", ""]
    lines.append(
        f"本轮选择 `{args.selected}` 作为当前候选检查点。它证明诊断工具可以从 rollout 特征出发，"
        "指导算法、数据配比和检查点选择；但该结果仍需不同 seed 复现，并满足业务 precision 下限后才能验收。"
    )
    lines.extend(["", "## 独立评测", "", "| 阶段 | Precision | Recall | F1 | F2 | FN | FP | Accuracy |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in metric_rows:
        lines.append(
            "| {name} | {precision:.6f} | {recall:.6f} | {f1:.6f} | {f2:.6f} | "
            "{false_negative} | {false_positive} | {accuracy:.6f} |".format(**row)
        )
    lines.extend(["", "## 观察到的特征", ""])
    lines.extend(f"- {item}" for item in args.observation)
    lines.extend(["", "## 优化决策", ""])
    lines.extend(f"- {item}" for item in args.decision)
    if rollout_steps:
        lines.extend(["", "## Rollout 证据", "", "| Step | Reward | BANKRUPTCY | HEALTHY | Effective Group | Constant Group |", "|---:|---:|---:|---:|---:|---:|"])
        for step in rollout_steps:
            counts = step.get("prediction_counts", {})
            lines.append(
                f"| {step.get('step')} | {format_number(step.get('reward_mean'))} | "
                f"{counts.get('BANKRUPTCY', 0)} | {counts.get('HEALTHY', 0)} | "
                f"{format_number(step.get('effective_group_rate'))} | {format_number(step.get('constant_group_rate'))} |"
            )
    if args.training_step:
        lines.extend(["", "## 训练稳定性", "", "| Step | Reward | Advantage | Entropy | KL Loss | Grad Norm | Clip Fraction | Response Clip |", "|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for step in args.training_step:
            lines.append(
                "| {step} | {reward_mean:.6f} | {advantage_mean:.6f} | {entropy:.6f} | "
                "{kl_loss:.6f} | {grad_norm:.6f} | {clip_fraction:.6f} | {response_clip_ratio:.6f} |".format(**step)
            )
    lines.extend(["", "## 限制与下一步", ""])
    lines.extend(f"- {item}" for item in args.limitation)
    lines.extend(["", "## 原始证据索引", ""])
    for row in metric_rows:
        lines.append(f"- `{row['name']}`: `{row['metrics_path']}`")
    if args.run_manifest:
        lines.append(f"- 运行清单: `{args.run_manifest.resolve()}`")
    if args.rollout_summary:
        lines.append(f"- Rollout 汇总: `{args.rollout_summary.resolve()}`")
    (args.output / "case_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
