"""Step-level live monitoring helpers for terminal and Jupyter environments."""

from __future__ import annotations

import json
import re
import statistics
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


def build_snapshot(
    log_path: str | Path,
    rollout_dir: str | Path,
    extreme_ratio_threshold: float = 0.75,
) -> dict:
    metric_rows = {row["step"]: row for row in read_step_metrics(log_path)}
    rollout_steps = read_rollout_steps(rollout_dir)
    steps = []
    for step in sorted(set(metric_rows) | set(rollout_steps)):
        metric = metric_rows.get(step, {})
        records = rollout_steps.get(step, [])
        correctness_groups = defaultdict(list)
        reward_groups = defaultdict(list)
        for record in records:
            correctness = record.get("acc")
            group_id = str(record.get("uid", record.get("group_id", "")))
            if isinstance(correctness, bool):
                correctness_groups[group_id].append(correctness)
            score = record.get("score", record.get("reward"))
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                reward_groups[group_id].append(float(score) > 0)
        correctness_types = Counter()
        for correctness_values in correctness_groups.values():
            ratio = sum(correctness_values) / len(correctness_values)
            if correctness_values and all(correctness_values):
                correctness_types["all_correct"] += 1
            elif correctness_values and not any(correctness_values):
                correctness_types["all_wrong"] += 1
            elif ratio >= extreme_ratio_threshold:
                correctness_types["mostly_correct"] += 1
            elif ratio <= 1.0 - extreme_ratio_threshold:
                correctness_types["mostly_wrong"] += 1
            else:
                correctness_types["mixed"] += 1
        reward_types = Counter()
        for outcomes in reward_groups.values():
            ratio = sum(outcomes) / len(outcomes)
            if ratio == 1.0:
                reward_types["all_positive"] += 1
            elif ratio == 0.0:
                reward_types["all_non_positive"] += 1
            elif ratio >= extreme_ratio_threshold:
                reward_types["mostly_positive"] += 1
            elif ratio <= 1.0 - extreme_ratio_threshold:
                reward_types["mostly_non_positive"] += 1
            else:
                reward_types["mixed"] += 1
        has_predictions = any("pred" in record for record in records)
        if has_predictions:
            valid = sum(record.get("pred") not in (None, "[INVALID]") for record in records)
        else:
            valid = sum(bool(str(record.get("output", "")).strip()) for record in records)
        positive = sum(float(record.get("score", 0)) > 0 for record in records)
        advantages = [float(record["advantage"]) for record in records if record.get("advantage") is not None]
        rewards = [
            float(record.get("score", record.get("reward")))
            for record in records
            if isinstance(record.get("score", record.get("reward")), (int, float))
            and not isinstance(record.get("score", record.get("reward")), bool)
        ]
        steps.append({
            "step": step,
            "rollouts": len(records),
            "valid_ratio": valid / len(records) if records else None,
            "positive_ratio": positive / len(records) if records else None,
            "has_correctness": bool(correctness_groups),
            "all_correct_groups": correctness_types["all_correct"],
            "all_wrong_groups": correctness_types["all_wrong"],
            "mostly_correct_groups": correctness_types["mostly_correct"],
            "mostly_wrong_groups": correctness_types["mostly_wrong"],
            "mixed_groups": correctness_types["mixed"],
            "all_positive_groups": reward_types["all_positive"],
            "all_non_positive_groups": reward_types["all_non_positive"],
            "mostly_positive_groups": reward_types["mostly_positive"],
            "mostly_non_positive_groups": reward_types["mostly_non_positive"],
            "reward_mean": metric.get("critic/score/mean"),
            "reward_min": min(rewards) if rewards else metric.get("critic/score/min"),
            "reward_max": max(rewards) if rewards else metric.get("critic/score/max"),
            "reward_std": statistics.pstdev(rewards) if len(rewards) > 1 else (0.0 if rewards else None),
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
