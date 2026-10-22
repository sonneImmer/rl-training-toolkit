#!/usr/bin/env python3
"""Summarize group-level AMLSim rollout learning signals."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def classify(exact: int, size: int) -> str:
    if exact == size:
        return "all_correct"
    if exact == 0:
        return "all_wrong"
    if exact >= size - 1:
        return "near_all_correct"
    if exact <= 1:
        return "near_all_wrong"
    return "mixed"


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[row["input"]].append(row)
    labels = Counter()
    gradient_groups = 0
    expected = Counter()
    per_typology = defaultdict(lambda: {"groups": 0, "rows": 0, "exact": 0, "reward_sum": 0.0, "gradient_groups": 0})
    for items in groups.values():
        exact = sum(bool(item.get("acc")) for item in items)
        label = classify(exact, len(items))
        labels[label] += 1
        has_gradient = len({round(float(item["score"]), 8) for item in items}) > 1
        gradient_groups += has_gradient
        typology = items[0].get("expected", "UNKNOWN")
        expected[typology] += 1
        bucket = per_typology[typology]
        bucket["groups"] += 1
        bucket["rows"] += len(items)
        bucket["exact"] += exact
        bucket["reward_sum"] += sum(float(item["score"]) for item in items)
        bucket["gradient_groups"] += has_gradient
        bucket[label] = bucket.get(label, 0) + 1
    type_summary = {}
    for typology, bucket in sorted(per_typology.items()):
        type_summary[typology] = {
            "groups": bucket["groups"],
            "exact_rate": bucket["exact"] / bucket["rows"],
            "mean_reward": bucket["reward_sum"] / bucket["rows"],
            "gradient_group_rate": bucket["gradient_groups"] / bucket["groups"],
            "all_correct": bucket.get("all_correct", 0),
            "all_wrong": bucket.get("all_wrong", 0),
            "near_all_correct": bucket.get("near_all_correct", 0),
            "near_all_wrong": bucket.get("near_all_wrong", 0),
            "mixed": bucket.get("mixed", 0),
        }
    return {
        "step": int(rows[0]["step"]),
        "rows": len(rows),
        "groups": len(groups),
        "exact_rate": sum(bool(row.get("acc")) for row in rows) / len(rows),
        "format_valid_rate": sum(bool(row.get("format_valid")) for row in rows) / len(rows),
        "mean_reward": statistics.mean(float(row["score"]) for row in rows),
        "min_reward": min(float(row["score"]) for row in rows),
        "max_reward": max(float(row["score"]) for row in rows),
        "gradient_groups": gradient_groups,
        "gradient_group_rate": gradient_groups / len(groups),
        **labels,
        "expected_groups": dict(expected),
        "per_typology": type_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = [summarize(path) for path in sorted(args.rollouts.glob("*.jsonl"), key=lambda p: int(p.stem))]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rollout_summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    columns = [key for key in summaries[0] if key not in {"expected_groups", "per_typology"}]
    with (args.output / "rollout_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key, 0) for key in columns})
    with (args.output / "rollout_typology_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["step", "typology", "groups", "exact_rate", "mean_reward", "gradient_group_rate", "all_correct", "all_wrong", "near_all_correct", "near_all_wrong", "mixed"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for summary in summaries:
            for typology, values in summary["per_typology"].items():
                writer.writerow({"step": summary["step"], "typology": typology, **values})
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
