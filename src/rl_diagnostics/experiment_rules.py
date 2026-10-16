"""Versioned, evidence-only labels for live RL experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Optional


RULE_VERSION = "1.0.0"


@dataclass
class RuleConfig:
    window_steps: int = 2
    truncation_threshold: float = 0.20
    valid_ratio_threshold: float = 0.80
    low_signal_ratio_threshold: float = 0.80
    gradient_spike_multiplier: float = 3.0
    kl_drift_multiplier: float = 3.0


def _values(rows: list[dict], key: str) -> list[tuple[int, float]]:
    return [
        (int(row["step"]), float(row[key]))
        for row in rows
        if row.get(key) is not None
    ]


def _window_label(
    rows: list[dict],
    key: str,
    threshold: float,
    comparison: str,
    name: str,
    title: str,
    severity: str,
    config: RuleConfig,
) -> Optional[dict]:
    values = _values(rows, key)
    if len(values) < config.window_steps:
        return None
    window = values[-config.window_steps:]
    if comparison == "gte":
        fired = all(value >= threshold for _, value in window)
    else:
        fired = all(value <= threshold for _, value in window)
    if not fired:
        return None
    return {
        "name": name,
        "title": title,
        "severity": severity,
        "status": "active",
        "first_step": window[0][0],
        "last_step": window[-1][0],
        "observed_value": window[-1][1],
        "threshold": threshold,
        "metric": key,
        "evidence_steps": [step for step, _ in window],
        "rule_version": RULE_VERSION,
    }


def _low_signal_rows(rows: list[dict]) -> list[dict]:
    enriched = []
    for row in rows:
        if row.get("has_correctness"):
            total = sum(int(row.get(key, 0)) for key in (
                "all_correct_groups", "mostly_correct_groups", "mixed_groups",
                "mostly_wrong_groups", "all_wrong_groups",
            ))
            constant = int(row.get("all_correct_groups", 0)) + int(row.get("all_wrong_groups", 0))
        else:
            total = sum(int(row.get(key, 0)) for key in (
                "all_positive_groups", "mostly_positive_groups",
                "mostly_non_positive_groups", "all_non_positive_groups",
            ))
            constant = int(row.get("all_positive_groups", 0)) + int(row.get("all_non_positive_groups", 0))
        item = dict(row)
        item["constant_group_ratio"] = constant / total if total else None
        enriched.append(item)
    return enriched


def evaluate_experiment_rules(
    rows: list[dict],
    status: str,
    config: Optional[RuleConfig] = None,
) -> list[dict]:
    """Return active labels. Labels describe evidence and never infer causes."""
    config = config or RuleConfig()
    labels: list[dict] = []

    for args in (
        ("clip_ratio", config.truncation_threshold, "gte", "high_truncation",
         "输出截断偏高", "warning"),
        ("valid_ratio", config.valid_ratio_threshold, "lte", "low_valid_ratio",
         "有效回答比例偏低", "warning"),
    ):
        label = _window_label(rows, *args, config)
        if label:
            labels.append(label)

    signal_label = _window_label(
        _low_signal_rows(rows), "constant_group_ratio",
        config.low_signal_ratio_threshold, "gte",
        "low_group_signal", "组内奖励区分信号不足", "warning", config,
    )
    if signal_label:
        labels.append(signal_label)

    grad_values = _values(rows, "grad_norm")
    if len(grad_values) >= 4:
        baseline_values = [value for _, value in grad_values[:-1]]
        baseline = median(baseline_values)
        step, latest = grad_values[-1]
        if baseline > 0 and latest >= baseline * config.gradient_spike_multiplier:
            labels.append({
                "name": "gradient_spike",
                "title": "梯度范数突增",
                "severity": "critical",
                "status": "active",
                "first_step": step,
                "last_step": step,
                "observed_value": latest,
                "threshold": baseline * config.gradient_spike_multiplier,
                "baseline_value": baseline,
                "metric": "grad_norm",
                "evidence_steps": [step],
                "rule_version": RULE_VERSION,
            })

    kl_values = _values(rows, "kl_loss")
    if len(kl_values) >= 6:
        baseline = median(value for _, value in kl_values[:-3])
        tail = kl_values[-3:]
        tail_values = [value for _, value in tail]
        if baseline > 0 and tail_values == sorted(tail_values) and tail_values[-1] >= baseline * config.kl_drift_multiplier:
            labels.append({
                "name": "kl_drift",
                "title": "KL 连续上升",
                "severity": "warning",
                "status": "active",
                "first_step": tail[0][0],
                "last_step": tail[-1][0],
                "observed_value": tail_values[-1],
                "threshold": baseline * config.kl_drift_multiplier,
                "baseline_value": baseline,
                "metric": "kl_loss",
                "evidence_steps": [step for step, _ in tail],
                "rule_version": RULE_VERSION,
            })

    if rows and rows[-1].get("rollouts", 0) == 0 and any(row.get("rollouts", 0) for row in rows[:-1]):
        step = int(rows[-1]["step"])
        labels.append({
            "name": "rollout_missing",
            "title": "最新训练步缺少 rollout",
            "severity": "critical",
            "status": "active",
            "first_step": step,
            "last_step": step,
            "observed_value": 0,
            "threshold": 1,
            "metric": "rollouts",
            "evidence_steps": [step],
            "rule_version": RULE_VERSION,
        })

    if status in ("failed", "stopped"):
        labels.append({
            "name": f"experiment_{status}",
            "title": "训练失败" if status == "failed" else "训练已停止更新",
            "severity": "critical" if status == "failed" else "warning",
            "status": "active",
            "first_step": rows[-1]["step"] if rows else None,
            "last_step": rows[-1]["step"] if rows else None,
            "metric": "experiment_status",
            "evidence_steps": [rows[-1]["step"]] if rows else [],
            "rule_version": RULE_VERSION,
        })

    if status == "completed_with_errors":
        labels.append({
            "name": "post_training_error",
            "title": "训练完成后出现尾部错误",
            "severity": "warning",
            "status": "active",
            "first_step": rows[-1]["step"] if rows else None,
            "last_step": rows[-1]["step"] if rows else None,
            "metric": "experiment_status",
            "evidence_steps": [rows[-1]["step"]] if rows else [],
            "rule_version": RULE_VERSION,
        })

    return labels


def rule_config_dict(config: Optional[RuleConfig] = None) -> dict:
    return asdict(config or RuleConfig())
