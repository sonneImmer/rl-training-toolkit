"""Versioned, evidence-only labels for live RL experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Optional


RULE_VERSION = "1.1.0"


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


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


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


def _matching_rows(rows: list[dict], steps: list[int]) -> list[dict]:
    step_set = set(steps)
    return [row for row in rows if int(row.get("step", -1)) in step_set]


def _explain_label(label: dict, rows: list[dict], error_summary: Optional[str]) -> dict:
    evidence_rows = _matching_rows(rows, label.get("evidence_steps", []))
    name = label["name"]
    threshold = float(label.get("threshold", 0.0))
    suggestions: list[str] = []
    evidence: list[dict] = []

    if name == "low_group_signal":
        parts = []
        for row in evidence_rows:
            denominator = int(row.get("analyzable_groups", 0))
            numerator = int(row.get("constant_score_groups", 0)) + int(row.get("near_constant_groups", 0))
            ratio = float(row.get("low_information_group_ratio", 0.0))
            parts.append(f"Step {row['step']} 低信息 Group {numerator}/{denominator}（{_percent(ratio)}）")
            evidence.append({"step": row["step"], "value": ratio, "numerator": numerator, "denominator": denominator})
        label.update({
            "observation": "；".join(parts),
            "decision_reason": f"连续 {len(evidence_rows)} 个 Step 的低信息 Group 比例均达到 {_percent(threshold)} 判断线。",
            "meaning": "同一 Prompt 的多次 Rollout 得分相同或接近，组内相对比较可提供的策略学习信号较少。",
            "boundary": "这不等于整个训练没有梯度，也不表示题目一定全对或全错；KL 等损失仍可能产生梯度。",
            "suggestions": ["打开证据 Step，检查同组 Reward 与 Advantage 是否集中为相同值。", "再核对 Reward 是否区分了答案质量，以及 Rollout 数量是否足够。"],
        })
    elif name == "high_truncation":
        parts = [f"Step {row['step']} 截断率 {_percent(float(row['clip_ratio']))}" for row in evidence_rows]
        evidence = [{"step": row["step"], "value": row["clip_ratio"]} for row in evidence_rows]
        label.update({
            "observation": "；".join(parts),
            "decision_reason": f"连续 {len(evidence_rows)} 个 Step 的截断率均达到 {_percent(threshold)} 判断线。",
            "meaning": "较多回答在最大生成长度处结束，答案或推理过程可能没有完整输出。",
            "boundary": "截断不直接证明答案错误；仍需查看原始输出末尾和最大生成长度设置。",
            "suggestions": ["打开证据 Step 查看输出是否在句中终止。", "检查 response_length、EOS 与输出格式约束。"],
        })
    elif name == "low_valid_ratio":
        parts = [f"Step {row['step']} 有效回答比例 {_percent(float(row['valid_ratio']))}" for row in evidence_rows]
        evidence = [{"step": row["step"], "value": row["valid_ratio"]} for row in evidence_rows]
        label.update({
            "observation": "；".join(parts),
            "decision_reason": f"连续 {len(evidence_rows)} 个 Step 的有效回答比例均不高于 {_percent(threshold)}。",
            "meaning": "解析器或质量规则判定为可用的回答不足，进入 Reward 与训练分析的有效样本减少。",
            "boundary": "这不能单独区分是模型输出差、格式要求不合理，还是解析规则不兼容。",
            "suggestions": ["查看证据 Step 中不可用回答的质量状态。", "对照 Prompt 格式、解析器规则和 Reward 输入字段。"],
        })
    elif name == "gradient_spike":
        baseline = float(label.get("baseline_value", 0.0))
        current = float(label.get("observed_value", 0.0))
        evidence = [{"step": label.get("last_step"), "value": current}]
        label.update({
            "observation": f"Step {label.get('last_step')} Grad Norm 为 {current:.4g}，此前中位数为 {baseline:.4g}。",
            "decision_reason": f"当前值达到此前中位数的 {current / baseline:.1f} 倍，超过 {label.get('threshold', 0):.4g} 判断线。" if baseline else "当前梯度范数超过配置判断线。",
            "meaning": "本步整体参数更新强度明显高于此前水平，需要检查是否为异常尖峰。",
            "boundary": "Grad Norm 是所有参与反向传播的样本和 Token 汇总值，不能反推出具体哪个 Group 导致尖峰。",
            "suggestions": ["联看同 Step 的 Reward、KL、Clip Fraction 和低信息 Group。", "若连续出现，再检查学习率、异常 Reward 和梯度裁剪。"],
        })
    elif name == "kl_drift":
        values = [float(row["kl_loss"]) for row in evidence_rows]
        baseline = float(label.get("baseline_value", 0.0))
        evidence = [{"step": row["step"], "value": row["kl_loss"]} for row in evidence_rows]
        label.update({
            "observation": "；".join(f"Step {row['step']} KL {float(row['kl_loss']):.4g}" for row in evidence_rows),
            "decision_reason": f"KL 连续上升，最新值为基线中位数的 {values[-1] / baseline:.1f} 倍。" if values and baseline else "KL 连续上升并超过配置判断线。",
            "meaning": "当前策略与参考模型的偏离持续扩大，KL 惩罚对优化目标的影响正在增强。",
            "boundary": "KL 上升不等于训练必然崩溃，也不能单独判断任务效果变好或变差。",
            "suggestions": ["联看独立评测、Reward 和 Entropy 是否同步变化。", "检查 KL 系数、学习率及是否存在异常高 Advantage。"],
        })
    elif name == "rollout_missing":
        evidence = [{"step": label.get("last_step"), "value": 0}]
        label.update({
            "observation": f"Step {label.get('last_step')} 没有读取到 Rollout，而此前 Step 存在 Rollout。",
            "decision_reason": "最新训练步的 Rollout 数量为 0。",
            "meaning": "该 Step 无法进行回答质量、组内信号和原始证据分析。",
            "boundary": "这可能是落盘延迟、文件路径变化或训练未生成，不能仅凭此认定训练失败。",
            "suggestions": ["检查 Rollout 文件是否延迟写入或写到其他目录。", "核对训练日志中的当前 Step 与落盘文件名。"],
        })
    elif name == "experiment_stopped":
        step = label.get("last_step")
        label.update({
            "observation": f"日志和 Rollout 已停止更新，最后读取到的训练 Step 为 {step if step is not None else '未知'}。",
            "decision_reason": "实验目录超过配置的无更新时长，且日志中没有明确完成标记。",
            "meaning": "监控无法确认训练仍在继续，需要检查进程或任务状态。",
            "boundary": "停止更新可能来自任务结束、进程退出或落盘中断，不自动等于训练失败。",
            "suggestions": ["检查训练进程和任务平台状态。", "确认日志、Rollout 与 Checkpoint 的最后更新时间。"],
        })
    elif name in ("experiment_failed", "post_training_error"):
        status_text = error_summary or "日志尾部未提取到更具体的错误行。"
        label.update({
            "observation": status_text,
            "decision_reason": "日志检测到错误标记，并结合训练进度判定为失败或完成后尾部错误。",
            "meaning": "训练结果需要结合最后有效 Step 和错误位置判断是否可用。",
            "boundary": "状态标签不自动判断已生成 Checkpoint 或 Rollout 是否损坏。",
            "suggestions": ["查看日志尾部错误及最后一个成功 Step。", "确认 Checkpoint、Rollout 和独立评测产物是否完整。"],
        })

    label["evidence"] = evidence
    label.setdefault("observation", "已满足本地规则的触发条件。")
    label.setdefault("decision_reason", "指标达到配置判断线。")
    label.setdefault("meaning", "该标签仅描述训练中可观察到的现象。")
    label.setdefault("boundary", "不能仅凭单个标签确定根因或实际任务效果。")
    label.setdefault("suggestions", suggestions)
    return label


def evaluate_experiment_rules(
    rows: list[dict],
    status: str,
    config: Optional[RuleConfig] = None,
    error_summary: Optional[str] = None,
) -> list[dict]:
    """Return active labels. Labels describe evidence and never infer causes."""
    config = config or RuleConfig()
    labels: list[dict] = []

    for args in (
        ("clip_ratio", config.truncation_threshold, "gte", "high_truncation", "输出截断偏高", "warning"),
        ("valid_ratio", config.valid_ratio_threshold, "lte", "low_valid_ratio", "有效回答比例偏低", "warning"),
        ("low_information_group_ratio", config.low_signal_ratio_threshold, "gte", "low_group_signal", "组内奖励区分信号不足", "warning"),
    ):
        label = _window_label(rows, *args, config)
        if label:
            labels.append(label)

    grad_values = _values(rows, "grad_norm")
    if len(grad_values) >= 4:
        baseline_values = [value for _, value in grad_values[:-1]]
        baseline = median(baseline_values)
        step, latest = grad_values[-1]
        if baseline > 0 and latest >= baseline * config.gradient_spike_multiplier:
            labels.append({
                "name": "gradient_spike", "title": "梯度范数突增", "severity": "critical", "status": "active",
                "first_step": step, "last_step": step, "observed_value": latest,
                "threshold": baseline * config.gradient_spike_multiplier, "baseline_value": baseline,
                "metric": "grad_norm", "evidence_steps": [step], "rule_version": RULE_VERSION,
            })

    kl_values = _values(rows, "kl_loss")
    if len(kl_values) >= 6:
        baseline = median(value for _, value in kl_values[:-3])
        tail = kl_values[-3:]
        tail_values = [value for _, value in tail]
        if baseline > 0 and tail_values == sorted(tail_values) and tail_values[-1] >= baseline * config.kl_drift_multiplier:
            labels.append({
                "name": "kl_drift", "title": "KL 连续上升", "severity": "warning", "status": "active",
                "first_step": tail[0][0], "last_step": tail[-1][0], "observed_value": tail_values[-1],
                "threshold": baseline * config.kl_drift_multiplier, "baseline_value": baseline,
                "metric": "kl_loss", "evidence_steps": [step for step, _ in tail], "rule_version": RULE_VERSION,
            })

    if rows and rows[-1].get("rollouts", 0) == 0 and any(row.get("rollouts", 0) for row in rows[:-1]):
        step = int(rows[-1]["step"])
        labels.append({
            "name": "rollout_missing", "title": "最新训练步缺少 Rollout", "severity": "critical", "status": "active",
            "first_step": step, "last_step": step, "observed_value": 0, "threshold": 1,
            "metric": "rollouts", "evidence_steps": [step], "rule_version": RULE_VERSION,
        })

    if status in ("failed", "stopped"):
        labels.append({
            "name": f"experiment_{status}", "title": "训练失败" if status == "failed" else "训练已停止更新",
            "severity": "critical" if status == "failed" else "warning", "status": "active",
            "first_step": rows[-1]["step"] if rows else None, "last_step": rows[-1]["step"] if rows else None,
            "metric": "experiment_status", "evidence_steps": [rows[-1]["step"]] if rows else [], "rule_version": RULE_VERSION,
        })

    if status == "completed_with_errors":
        labels.append({
            "name": "post_training_error", "title": "训练完成后出现尾部错误", "severity": "warning", "status": "active",
            "first_step": rows[-1]["step"] if rows else None, "last_step": rows[-1]["step"] if rows else None,
            "metric": "experiment_status", "evidence_steps": [rows[-1]["step"]] if rows else [], "rule_version": RULE_VERSION,
        })

    return [_explain_label(label, rows, error_summary) for label in labels]


def rule_config_dict(config: Optional[RuleConfig] = None) -> dict:
    return asdict(config or RuleConfig())
