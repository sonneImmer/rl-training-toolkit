"""REQ-07: Training stability diagnosis.

Detects anomalous trends in training metrics and outputs
evidence-based alerts with severity and investigation suggestions.
First-round only: evidence alerts, no automatic root-cause attribution.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Optional

from ..schemas.data_models import StepMetric
from ..config.stability_config import StabilityConfig, StabilityRuleConfig
from .group_signal import StepAggregation


@dataclass
class StabilityAlert:
    """A single stability alert with evidence."""
    rule_name: str
    severity: str  # warning | critical | info
    metric_name: str
    trigger_value: float
    baseline_value: float
    threshold: float
    window_steps: list[int] = field(default_factory=list)
    phase: str = "train"
    description: str = ""
    investigation_hints: list[str] = field(default_factory=list)
    config_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class StabilityReport:
    """Aggregated stability report."""
    config_version: str = ""
    total_alerts: int = 0
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    alerts: list[StabilityAlert] = field(default_factory=list)
    rules_checked: list[str] = field(default_factory=list)
    rules_skipped: list[str] = field(default_factory=list)
    metrics_available: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


# ── Helpers ──────────────────────────────────────────────────────

def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return statistics.median(values)


def _mad(values: list[float], center: Optional[float] = None) -> float:
    """Median Absolute Deviation."""
    if not values:
        return 0.0
    c = center if center is not None else _median(values)
    return _median([abs(v - c) for v in values])


def _group_metrics_by_phase(
    metrics: list[StepMetric], metric_names: list[str],
    phase: str = "train",
) -> dict[str, list[tuple[int, float]]]:
    """Group metric values by metric name for a given phase.

    Returns {metric_name: [(step, value), ...]} sorted by step.
    """
    result: dict[str, list[tuple[int, float]]] = {}
    for m in metrics:
        if m.phase != phase:
            continue
        if m.metric_name in metric_names or not metric_names:
            result.setdefault(m.metric_name, []).append((m.step, m.metric_value))
    for name in result:
        result[name].sort(key=lambda x: x[0])
    return result


def _extract_series(
    metrics: list[StepMetric], metric_name: str, phase: str = "train",
) -> list[tuple[int, float]]:
    """Extract (step, value) pairs for a single metric in a phase."""
    series = [(m.step, m.metric_value) for m in metrics
              if m.metric_name == metric_name and m.phase == phase]
    series.sort(key=lambda x: x[0])
    return series


def _window_values(series: list[tuple[int, float]], start: int, end: int) -> list[float]:
    """Extract values in a step range [start, end)."""
    return [v for s, v in series if start <= s < end]


def _pct_change(current: float, baseline: float) -> float:
    """Percentage change, safe against zero baseline."""
    if abs(baseline) < 1e-12:
        return 0.0 if abs(current) < 1e-12 else float("inf")
    return (current - baseline) / abs(baseline)


# ── Rule implementations ─────────────────────────────────────────

def _check_explore_collapse(
    metrics: list[StepMetric], cfg: StabilityRuleConfig, config_version: str,
) -> list[StabilityAlert]:
    """Rule: actor/entropy median drops >50% vs baseline window."""
    alerts = []
    if not cfg.enabled:
        return alerts
    names = cfg.extra.get("metric_names", ["actor/entropy"])
    for phase in ("train",):
        for name in names:
            series = _extract_series(metrics, name, phase)
            if len(series) < cfg.baseline_window + cfg.window_size:
                continue
            baseline_vals = [v for _, v in series[:cfg.baseline_window]]
            window_vals = [v for _, v in series[-cfg.window_size:]]
            baseline_med = _median(baseline_vals)
            window_med = _median(window_vals)
            if abs(baseline_med) < 1e-12:
                continue
            drop_pct = abs(baseline_med - window_med) / abs(baseline_med)
            if drop_pct > cfg.threshold:
                window_steps = [s for s, _ in series[-cfg.window_size:]]
                alerts.append(StabilityAlert(
                    rule_name="explore_collapse",
                    severity="critical" if drop_pct > 2 * cfg.threshold else "warning",
                    metric_name=name,
                    trigger_value=window_med,
                    baseline_value=baseline_med,
                    threshold=cfg.threshold,
                    window_steps=window_steps,
                    phase=phase,
                    description=(
                        f"Entropy median dropped {drop_pct:.1%} from baseline "
                        f"{baseline_med:.6f} to {window_med:.6f}"
                    ),
                    investigation_hints=[
                        "Check entropy coefficient scheduling",
                        "Verify temperature settings",
                        "Check for data repetition or reward dominance",
                    ],
                    config_version=config_version,
                ))
    return alerts


def _check_kl_drift(
    metrics: list[StepMetric], cfg: StabilityRuleConfig, config_version: str,
) -> list[StabilityAlert]:
    """Rule: |KL| median > 3x baseline for at least 2 consecutive windows."""
    alerts = []
    if not cfg.enabled:
        return alerts
    names = cfg.extra.get("metric_names", ["actor/ppo_kl", "actor/kl_loss"])
    for phase in ("train",):
        for name in names:
            series = _extract_series(metrics, name, phase)
            if len(series) < cfg.baseline_window + cfg.window_size * cfg.min_windows:
                continue
            baseline_vals = [abs(v) for _, v in series[:cfg.baseline_window]]
            baseline_med = _median(baseline_vals)
            if baseline_med < 1e-12:
                continue
            consecutive = 0
            first_window_steps: list[int] = []
            total_windows = (len(series) - cfg.baseline_window) // cfg.window_size
            for w in range(total_windows):
                start_idx = cfg.baseline_window + w * cfg.window_size
                end_idx = start_idx + cfg.window_size
                window_vals = [abs(v) for _, v in series[start_idx:end_idx]]
                if not window_vals:
                    continue
                window_med = _median(window_vals)
                if window_med > baseline_med * cfg.threshold:
                    consecutive += 1
                    if consecutive == 1:
                        first_window_steps = [s for s, _ in series[start_idx:end_idx]]
                    if consecutive >= cfg.min_windows:
                        alerts.append(StabilityAlert(
                            rule_name="kl_drift",
                            severity="critical",
                            metric_name=name,
                            trigger_value=window_med,
                            baseline_value=baseline_med,
                            threshold=cfg.threshold,
                            window_steps=first_window_steps,
                            phase=phase,
                            description=(
                                f"|KL| median {window_med:.6f} exceeds baseline "
                                f"{baseline_med:.6f} by {window_med / baseline_med:.1f}x "
                                f"for {consecutive} consecutive windows"
                            ),
                            investigation_hints=[
                                "Check learning rate scheduling",
                                "Verify KL coefficient settings",
                                "Review number of update steps per iteration",
                            ],
                            config_version=config_version,
                        ))
                        break
                else:
                    consecutive = 0
    return alerts


def _check_clip_saturation(
    metrics: list[StepMetric], cfg: StabilityRuleConfig, config_version: str,
) -> list[StabilityAlert]:
    """Rule: pg_clipfrac consecutive windows above threshold."""
    alerts = []
    if not cfg.enabled:
        return alerts
    names = cfg.extra.get("metric_names", ["actor/pg_clipfrac"])
    for phase in ("train",):
        for name in names:
            series = _extract_series(metrics, name, phase)
            if len(series) < cfg.window_size * cfg.min_windows:
                continue
            total_windows = len(series) // cfg.window_size
            consecutive = 0
            trigger_steps: list[int] = []
            for w in range(total_windows):
                start_idx = w * cfg.window_size
                end_idx = start_idx + cfg.window_size
                window_vals = [v for _, v in series[start_idx:end_idx]]
                if not window_vals:
                    continue
                window_med = _median(window_vals)
                if window_med > cfg.threshold:
                    consecutive += 1
                    if consecutive == 1:
                        trigger_steps = [s for s, _ in series[start_idx:end_idx]]
                    if consecutive >= cfg.min_windows:
                        alerts.append(StabilityAlert(
                            rule_name="clip_saturation",
                            severity="warning",
                            metric_name=name,
                            trigger_value=window_med,
                            baseline_value=0.0,
                            threshold=cfg.threshold,
                            window_steps=trigger_steps,
                            phase=phase,
                            description=(
                                f"Clip fraction median {window_med:.4f} exceeds "
                                f"threshold {cfg.threshold} for {consecutive} windows"
                            ),
                            investigation_hints=[
                                "Check learning rate",
                                "Review mini-batch size",
                                "Examine advantage scale",
                            ],
                            config_version=config_version,
                        ))
                        break
                else:
                    consecutive = 0
    return alerts


def _check_gradient_spike(
    metrics: list[StepMetric], cfg: StabilityRuleConfig, config_version: str,
) -> list[StabilityAlert]:
    """Rule: grad_norm exceeds rolling_median + N*MAD."""
    alerts = []
    if not cfg.enabled:
        return alerts
    names = cfg.extra.get("metric_names", ["actor/grad_norm"])
    mad_mult = cfg.extra.get("mad_multiplier", 5.0)
    for phase in ("train",):
        for name in names:
            series = _extract_series(metrics, name, phase)
            if len(series) < cfg.baseline_window + 1:
                continue
            for i in range(cfg.baseline_window, len(series)):
                baseline_vals = [v for _, v in series[i - cfg.baseline_window:i]]
                b_med = _median(baseline_vals)
                b_mad = _mad(baseline_vals, b_med)
                current_val = series[i][1]
                # When MAD is 0 (all values identical), use a small epsilon
                # so that any deviation from baseline can be detected
                effective_mad = b_mad if b_mad > 0 else abs(b_med) * 1e-6 + 1e-12
                threshold_val = b_med + mad_mult * effective_mad
                if current_val > threshold_val:
                    alerts.append(StabilityAlert(
                        rule_name="gradient_spike",
                        severity="warning",
                        metric_name=name,
                        trigger_value=current_val,
                        baseline_value=b_med,
                        threshold=threshold_val,
                        window_steps=[series[i][0]],
                        phase=phase,
                        description=(
                            f"Gradient norm {current_val:.4f} exceeds "
                            f"median+{mad_mult}*MAD ({threshold_val:.4f})"
                        ),
                        investigation_hints=[
                            "Check for anomalous batch data",
                            "Verify learning rate",
                            "Review reward scale",
                        ],
                        config_version=config_version,
                    ))
    return alerts


def _check_output_truncation(
    metrics: list[StepMetric], cfg: StabilityRuleConfig, config_version: str,
) -> list[StabilityAlert]:
    """Rule: response clip_ratio > threshold or rises > relative_rise_pp vs baseline."""
    alerts = []
    if not cfg.enabled:
        return alerts
    names = cfg.extra.get("metric_names", ["response_length/clip_ratio"])
    rise_pp = cfg.extra.get("relative_rise_pp", 10.0)
    for phase in ("train",):
        for name in names:
            series = _extract_series(metrics, name, phase)
            if len(series) < cfg.baseline_window + cfg.window_size:
                continue
            baseline_vals = [v for _, v in series[:cfg.baseline_window]]
            window_vals = [v for _, v in series[-cfg.window_size:]]
            baseline_med = _median(baseline_vals)
            window_med = _median(window_vals)
            window_steps = [s for s, _ in series[-cfg.window_size:]]
            triggered = False
            desc = ""
            if window_med > cfg.threshold:
                triggered = True
                desc = (
                    f"Output clip ratio {window_med:.4f} exceeds "
                    f"threshold {cfg.threshold}"
                )
            elif abs(baseline_med) > 1e-12:
                rise = (window_med - baseline_med) * 100  # percentage points
                if rise > rise_pp:
                    triggered = True
                    desc = (
                        f"Output clip ratio rose {rise:.1f}pp from "
                        f"baseline {baseline_med:.4f} to {window_med:.4f}"
                    )
            if triggered:
                alerts.append(StabilityAlert(
                    rule_name="output_truncation",
                    severity="warning",
                    metric_name=name,
                    trigger_value=window_med,
                    baseline_value=baseline_med,
                    threshold=cfg.threshold,
                    window_steps=window_steps,
                    phase=phase,
                    description=desc,
                    investigation_hints=[
                        "Check max response length setting",
                        "Review long reasoning chain patterns",
                        "Examine format requirements in prompts",
                    ],
                    config_version=config_version,
                ))
    return alerts


def _check_reward_signal_degradation(
    step_aggs: list[StepAggregation],
    cfg: StabilityRuleConfig, config_version: str,
) -> list[StabilityAlert]:
    """Rule: low_information_group_ratio > 0.80 or IPR persistently declining."""
    alerts = []
    if not cfg.enabled:
        return alerts
    if len(step_aggs) < cfg.baseline_window + cfg.window_size:
        return alerts
    low_info_key = cfg.extra.get("low_info_metric", "low_information_group_ratio")
    ipr_key = cfg.extra.get("ipr_metric", "mean_informative_pair_ratio")
    ipr_decline_pct = cfg.extra.get("ipr_decline_pct", 0.30)

    low_info_vals = [getattr(sa, low_info_key, 0.0) for sa in step_aggs]
    ipr_vals = [getattr(sa, ipr_key, 0.0) for sa in step_aggs]
    step_list = [sa.step for sa in step_aggs]

    window_low_info = low_info_vals[-cfg.window_size:]
    window_steps = step_list[-cfg.window_size:]
    window_med = _median(window_low_info)

    if window_med > cfg.threshold:
        alerts.append(StabilityAlert(
            rule_name="reward_signal_degradation",
            severity="critical",
            metric_name=low_info_key,
            trigger_value=window_med,
            baseline_value=0.0,
            threshold=cfg.threshold,
            window_steps=window_steps,
            phase="train",
            description=(
                f"Low-information group ratio {window_med:.2%} exceeds "
                f"threshold {cfg.threshold:.0%}"
            ),
            investigation_hints=[
                "Review data difficulty distribution",
                "Examine reward function design",
                "Consider algorithm choice (GRPO vs DAPO)",
            ],
            config_version=config_version,
        ))
        return alerts

    baseline_ipr = _median(ipr_vals[:cfg.baseline_window])
    window_ipr = _median(ipr_vals[-cfg.window_size:])
    if abs(baseline_ipr) > 1e-12:
        decline = (baseline_ipr - window_ipr) / abs(baseline_ipr)
        if decline > ipr_decline_pct:
            alerts.append(StabilityAlert(
                rule_name="reward_signal_degradation",
                severity="warning",
                metric_name=ipr_key,
                trigger_value=window_ipr,
                baseline_value=baseline_ipr,
                threshold=ipr_decline_pct,
                window_steps=window_steps,
                phase="train",
                description=(
                    f"Informative pair ratio declined {decline:.1%} from "
                    f"baseline {baseline_ipr:.4f} to {window_ipr:.4f}"
                ),
                investigation_hints=[
                    "Check if data difficulty is increasing over steps",
                    "Review reward function discrimination",
                    "Monitor for reward hacking or collapse",
                ],
                config_version=config_version,
            ))
    return alerts


def _check_performance_degradation(
    metrics: list[StepMetric], cfg: StabilityRuleConfig, config_version: str,
) -> list[StabilityAlert]:
    """Rule: throughput drops >20% vs same-phase baseline."""
    alerts = []
    if not cfg.enabled:
        return alerts
    names = cfg.extra.get("metric_names", ["perf/throughput"])
    for phase in ("train", "validation", "checkpoint"):
        for name in names:
            series = _extract_series(metrics, name, phase)
            if len(series) < cfg.baseline_window + cfg.window_size:
                continue
            baseline_vals = [v for _, v in series[:cfg.baseline_window]]
            window_vals = [v for _, v in series[-cfg.window_size:]]
            baseline_med = _median(baseline_vals)
            window_med = _median(window_vals)
            if abs(baseline_med) < 1e-12:
                continue
            drop = (baseline_med - window_med) / abs(baseline_med)
            if drop > cfg.threshold:
                window_steps = [s for s, _ in series[-cfg.window_size:]]
                alerts.append(StabilityAlert(
                    rule_name="performance_degradation",
                    severity="warning",
                    metric_name=name,
                    trigger_value=window_med,
                    baseline_value=baseline_med,
                    threshold=cfg.threshold,
                    window_steps=window_steps,
                    phase=phase,
                    description=(
                        f"Throughput dropped {drop:.1%} from baseline "
                        f"{baseline_med:.2f} to {window_med:.2f} (phase={phase})"
                    ),
                    investigation_hints=[
                        "Check generate/reward/update timing breakdown",
                        "Monitor communication overhead",
                        "Review checkpoint save frequency",
                    ],
                    config_version=config_version,
                ))
    return alerts


# ── Main entry point ─────────────────────────────────────────────

def run_stability_diagnosis(
    metrics: list[StepMetric],
    step_aggs: Optional[list[StepAggregation]] = None,
    config: Optional[StabilityConfig] = None,
) -> StabilityReport:
    """Run all stability rules and return a report.

    Args:
        metrics: StepMetric list from REQ-01 log parsing.
        step_aggs: StepAggregation list from REQ-03 group signal (optional).
        config: StabilityConfig with rule thresholds. Uses defaults if None.

    Returns:
        StabilityReport with alerts and metadata.
    """
    if config is None:
        config = StabilityConfig()

    report = StabilityReport(config_version=config.config_version)
    available_names = sorted(set(m.metric_name for m in metrics))
    report.metrics_available = available_names

    rule_configs = config.get_rule_configs()

    # Metric-based rules
    metric_rules = {
        "explore_collapse": _check_explore_collapse,
        "kl_drift": _check_kl_drift,
        "clip_saturation": _check_clip_saturation,
        "gradient_spike": _check_gradient_spike,
        "output_truncation": _check_output_truncation,
        "performance_degradation": _check_performance_degradation,
    }

    for rule_name, check_fn in metric_rules.items():
        cfg = rule_configs[rule_name]
        if cfg.enabled:
            alerts = check_fn(metrics, cfg, config.config_version)
            report.alerts.extend(alerts)
            report.rules_checked.append(rule_name)
        else:
            report.rules_skipped.append(rule_name)

    # Aggregation-based rule
    agg_cfg = rule_configs["reward_signal_degradation"]
    if agg_cfg.enabled and step_aggs:
        alerts = _check_reward_signal_degradation(step_aggs, agg_cfg, config.config_version)
        report.alerts.extend(alerts)
        report.rules_checked.append("reward_signal_degradation")
    elif agg_cfg.enabled and not step_aggs:
        report.rules_skipped.append("reward_signal_degradation (no step aggregations)")
    else:
        report.rules_skipped.append("reward_signal_degradation")

    # Summary
    report.total_alerts = len(report.alerts)
    report.critical_count = sum(1 for a in report.alerts if a.severity == "critical")
    report.warning_count = sum(1 for a in report.alerts if a.severity == "warning")
    report.info_count = sum(1 for a in report.alerts if a.severity == "info")

    return report
