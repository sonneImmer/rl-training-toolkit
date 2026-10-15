"""Step-level live monitoring helpers for terminal and Jupyter environments."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
METRIC = re.compile(r"([A-Za-z0-9_./-]+):([-+0-9.eE]+)")


def read_step_metrics(log_path: str | Path) -> list[dict]:
    path = Path(log_path)
    if not path.is_file():
        return []
    rows = {}
    for raw_line in path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw_line)
        match = re.search(r"step:(\d+) - ", line)
        if not match:
            continue
        row = {key: float(value) for key, value in METRIC.findall(line)}
        row["step"] = int(match.group(1))
        rows[row["step"]] = row
    return [rows[step] for step in sorted(rows)]


def read_rollout_steps(rollout_dir: str | Path) -> dict[int, list[dict]]:
    directory = Path(rollout_dir)
    result = {}
    if not directory.is_dir():
        return result
    for path in directory.glob("*.jsonl"):
        if not path.stem.isdigit():
            continue
        records = []
        for line in path.read_text(errors="replace").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        result[int(path.stem)] = records
    return result


def build_snapshot(log_path: str | Path, rollout_dir: str | Path) -> dict:
    metric_rows = {row["step"]: row for row in read_step_metrics(log_path)}
    rollout_steps = read_rollout_steps(rollout_dir)
    steps = []
    for step in sorted(set(metric_rows) | set(rollout_steps)):
        metric = metric_rows.get(step, {})
        records = rollout_steps.get(step, [])
        groups = defaultdict(list)
        for record in records:
            correctness = record.get("acc")
            if not isinstance(correctness, bool):
                correctness = float(record.get("score", 0)) > 0
            groups[str(record.get("uid", ""))].append(correctness)
        group_types = Counter()
        for correctness_values in groups.values():
            if correctness_values and all(correctness_values):
                group_types["all_correct"] += 1
            elif correctness_values and not any(correctness_values):
                group_types["all_wrong"] += 1
            else:
                group_types["mixed"] += 1
        has_predictions = any("pred" in record for record in records)
        if has_predictions:
            valid = sum(record.get("pred") not in (None, "[INVALID]") for record in records)
        else:
            valid = sum(bool(str(record.get("output", "")).strip()) for record in records)
        positive = sum(float(record.get("score", 0)) > 0 for record in records)
        advantages = [float(record["advantage"]) for record in records if record.get("advantage") is not None]
        steps.append({
            "step": step,
            "rollouts": len(records),
            "valid_ratio": valid / len(records) if records else None,
            "positive_ratio": positive / len(records) if records else None,
            "all_correct_groups": group_types["all_correct"],
            "all_wrong_groups": group_types["all_wrong"],
            "mixed_groups": group_types["mixed"],
            "reward_mean": metric.get("critic/score/mean"),
            "clip_ratio": metric.get("response_length/clip_ratio"),
            "entropy": metric.get("actor/entropy"),
            "kl_loss": metric.get("actor/kl_loss"),
            "grad_norm": metric.get("actor/grad_norm"),
            "pg_clipfrac": metric.get("actor/pg_clipfrac"),
            "advantage_mean": sum(advantages) / len(advantages) if advantages else metric.get("critic/advantages/mean"),
            "advantage_min": min(advantages) if advantages else metric.get("critic/advantages/min"),
            "advantage_max": max(advantages) if advantages else metric.get("critic/advantages/max"),
        })
    return {"latest_step": steps[-1]["step"] if steps else None, "steps": steps}


def latest_rollouts(rollout_dir: str | Path, limit: int = 8) -> list[dict]:
    steps = read_rollout_steps(rollout_dir)
    if not steps:
        return []
    return steps[max(steps)][:limit]
