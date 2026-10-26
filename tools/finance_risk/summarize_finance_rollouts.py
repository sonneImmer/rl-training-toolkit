#!/usr/bin/env python3
"""Summarize finance-risk rollout JSONL files into group-level diagnostic evidence."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def group_kind(scores: list[float]) -> str:
    correct = sum(score > 0.5 for score in scores)
    if correct == 0:
        return "all_wrong"
    if correct == len(scores):
        return "all_correct"
    if correct == 1:
        return "near_all_wrong"
    if correct == len(scores) - 1:
        return "near_all_correct"
    return "mixed"


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[row["uid"]].append(row)

    group_counts = Counter()
    label_group_counts = Counter()
    predictions = Counter()
    for items in groups.values():
        kind = group_kind([float(row["score"]) for row in items])
        label = items[0]["expected"]
        group_counts[kind] += 1
        label_group_counts[f"{label}:{kind}"] += 1
        predictions.update(row["pred"] for row in items)

    effective = sum(count for kind, count in group_counts.items() if kind not in {"all_wrong", "all_correct"})
    return {
        "step": int(rows[0]["step"]) if rows else None,
        "rows": len(rows),
        "groups": len(groups),
        "group_size_counts": dict(sorted(Counter(len(items) for items in groups.values()).items())),
        "reward_mean": sum(float(row["score"]) for row in rows) / len(rows) if rows else 0.0,
        "format_valid_rate": sum(bool(row.get("format_valid")) for row in rows) / len(rows) if rows else 0.0,
        "prediction_counts": dict(sorted(predictions.items())),
        "group_counts": dict(sorted(group_counts.items())),
        "label_group_counts": dict(sorted(label_group_counts.items())),
        "effective_group_rate": effective / len(groups) if groups else 0.0,
        "constant_group_rate": 1.0 - effective / len(groups) if groups else 0.0,
        "rules": [
            {
                "rule_id": "finance.constant_group_rate.high",
                "hit": bool(groups and (1.0 - effective / len(groups)) >= 0.5),
                "value": 1.0 - effective / len(groups) if groups else 0.0,
                "threshold": 0.5,
            },
            {
                "rule_id": "finance.positive_groups.no_advantage",
                "hit": all(
                    kind in {"all_wrong", "all_correct"}
                    for key, count in label_group_counts.items()
                    if count and key.startswith("BANKRUPTCY:")
                    for kind in [key.split(":", 1)[1]]
                ),
                "evidence": {
                    key: count for key, count in sorted(label_group_counts.items()) if key.startswith("BANKRUPTCY:")
                },
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    files = sorted(args.rollouts.glob("*.jsonl"), key=lambda path: int(path.stem))
    report = {"schema_version": "1.0.0", "steps": [summarize(path) for path in files]}
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
