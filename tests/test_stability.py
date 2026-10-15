"""Tests for REQ-07: Training stability diagnosis."""
import math
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_diagnostics.schemas.data_models import StepMetric
from rl_diagnostics.analyzers.group_signal import StepAggregation
from rl_diagnostics.analyzers.stability_analyzer import (
    run_stability_diagnosis, StabilityAlert, StabilityReport,
)
from rl_diagnostics.config.stability_config import StabilityConfig, STABILITY_CONFIG_VERSION


def _make_metric(step, name, value, phase="train"):
    return StepMetric(
        experiment_id="test_exp",
        step=step, epoch=step // 50, phase=phase,
        metric_name=name, metric_value=value,
    )


class TestExploreCollapse:
    """TC: entropy drops >50% triggers alert."""

    def test_no_collapse(self):
        metrics = [_make_metric(s, "actor/entropy", 2.0) for s in range(50)]
        report = run_stability_diagnosis(metrics)
        alerts = [a for a in report.alerts if a.rule_name == "explore_collapse"]
        assert len(alerts) == 0

    def test_collapse_detected(self):
        # Baseline entropy ~2.0 for 20 steps, then drops to 0.5 for 10 steps
        metrics = [_make_metric(s, "actor/entropy", 2.0) for s in range(20)]
        metrics += [_make_metric(s, "actor/entropy", 0.5) for s in range(20, 30)]
        report = run_stability_diagnosis(metrics)
        alerts = [a for a in report.alerts if a.rule_name == "explore_collapse"]
        assert len(alerts) >= 1
        assert alerts[0].severity in ("warning", "critical")
        assert alerts[0].baseline_value > 1.0
        assert alerts[0].trigger_value < 1.0

    def test_disabled_rule(self):
        config = StabilityConfig()
        config.explore_collapse.enabled = False
        metrics = [_make_metric(s, "actor/entropy", 2.0) for s in range(20)]
        metrics += [_make_metric(s, "actor/entropy", 0.1) for s in range(20, 30)]
        report = run_stability_diagnosis(metrics, config=config)
        assert "explore_collapse" in report.rules_skipped
        assert all(a.rule_name != "explore_collapse" for a in report.alerts)


class TestKLDrift:
    """TC: |KL| > 3x baseline for 2+ windows triggers alert."""

    def test_no_drift(self):
        metrics = [_make_metric(s, "actor/ppo_kl", 0.01) for s in range(60)]
        report = run_stability_diagnosis(metrics)
        alerts = [a for a in report.alerts if a.rule_name == "kl_drift"]
        assert len(alerts) == 0

    def test_drift_detected(self):
        # Baseline KL ~0.01 for 20 steps, then jumps to 0.1 for 20 steps
        metrics = [_make_metric(s, "actor/ppo_kl", 0.01) for s in range(20)]
        metrics += [_make_metric(s, "actor/ppo_kl", 0.1) for s in range(20, 40)]
        report = run_stability_diagnosis(metrics)
        alerts = [a for a in report.alerts if a.rule_name == "kl_drift"]
        assert len(alerts) >= 1
        assert alerts[0].severity == "critical"

    def test_single_window_not_triggered(self):
        """Single window spike should not trigger (min_windows=2)."""
        config = StabilityConfig()
        metrics = [_make_metric(s, "actor/ppo_kl", 0.01) for s in range(20)]
        # Only 10 steps of high KL (1 window, need 2)
        metrics += [_make_metric(s, "actor/ppo_kl", 0.1) for s in range(20, 30)]
        metrics += [_make_metric(s, "actor/ppo_kl", 0.01) for s in range(30, 40)]
        report = run_stability_diagnosis(metrics, config=config)
        kl_alerts = [a for a in report.alerts if a.rule_name == "kl_drift"]
        assert len(kl_alerts) == 0


class TestClipSaturation:
    """TC: pg_clipfrac above threshold for consecutive windows."""

    def test_no_saturation(self):
        metrics = [_make_metric(s, "actor/pg_clipfrac", 0.05) for s in range(60)]
        report = run_stability_diagnosis(metrics)
        alerts = [a for a in report.alerts if a.rule_name == "clip_saturation"]
        assert len(alerts) == 0

    def test_saturation_detected(self):
        # clipfrac at 0.3 for 30 steps (3 windows of 10)
        metrics = [_make_metric(s, "actor/pg_clipfrac", 0.3) for s in range(30)]
        report = run_stability_diagnosis(metrics)
        alerts = [a for a in report.alerts if a.rule_name == "clip_saturation"]
        assert len(alerts) >= 1


class TestGradientSpike:
    """TC: grad_norm exceeds median + 5*MAD."""

    def test_no_spike(self):
        metrics = [_make_metric(s, "actor/grad_norm", 1.0 + (s % 3) * 0.1) for s in range(40)]
        report = run_stability_diagnosis(metrics)
        alerts = [a for a in report.alerts if a.rule_name == "gradient_spike"]
        assert len(alerts) == 0

    def test_spike_detected(self):
        # Normal grad_norm ~1.0 for 30 steps, then a spike to 100
        metrics = [_make_metric(s, "actor/grad_norm", 1.0) for s in range(30)]
        metrics.append(_make_metric(30, "actor/grad_norm", 100.0))
        report = run_stability_diagnosis(metrics)
        alerts = [a for a in report.alerts if a.rule_name == "gradient_spike"]
        assert len(alerts) >= 1
        assert alerts[0].trigger_value > 50


class TestOutputTruncation:
    """TC: response clip_ratio above threshold or rising 10pp."""

    def test_no_truncation(self):
        metrics = [_make_metric(s, "response_length/clip_ratio", 0.05) for s in range(40)]
        report = run_stability_diagnosis(metrics)
        alerts = [a for a in report.alerts if a.rule_name == "output_truncation"]
        assert len(alerts) == 0

    def test_absolute_threshold(self):
        metrics = [_make_metric(s, "response_length/clip_ratio", 0.05) for s in range(20)]
        metrics += [_make_metric(s, "response_length/clip_ratio", 0.3) for s in range(20, 30)]
        report = run_stability_diagnosis(metrics)
        alerts = [a for a in report.alerts if a.rule_name == "output_truncation"]
        assert len(alerts) >= 1


class TestRewardSignalDegradation:
    """TC: low_information_group_ratio > 0.80 triggers alert."""

    def test_no_degradation(self):
        aggs = []
        for s in range(30):
            aggs.append(StepAggregation(
                step=s, epoch=0,
                total_groups=10, analyzable_groups=10,
                effective_groups=7, constant_score_groups=2,
                near_constant_groups=1,
                effective_group_ratio=0.7,
                low_information_group_ratio=0.3,
                mean_informative_pair_ratio=0.8,
            ))
        report = run_stability_diagnosis(metrics=[], step_aggs=aggs)
        alerts = [a for a in report.alerts if a.rule_name == "reward_signal_degradation"]
        assert len(alerts) == 0

    def test_degradation_detected(self):
        aggs = []
        for s in range(20):
            aggs.append(StepAggregation(
                step=s, epoch=0,
                total_groups=10, analyzable_groups=10,
                effective_groups=8, constant_score_groups=1,
                near_constant_groups=1,
                effective_group_ratio=0.8,
                low_information_group_ratio=0.2,
                mean_informative_pair_ratio=0.9,
            ))
        for s in range(20, 25):
            aggs.append(StepAggregation(
                step=s, epoch=0,
                total_groups=10, analyzable_groups=10,
                effective_groups=1, constant_score_groups=8,
                near_constant_groups=1,
                effective_group_ratio=0.1,
                low_information_group_ratio=0.9,
                mean_informative_pair_ratio=0.1,
            ))
        report = run_stability_diagnosis(metrics=[], step_aggs=aggs)
        alerts = [a for a in report.alerts if a.rule_name == "reward_signal_degradation"]
        assert len(alerts) >= 1
        assert alerts[0].severity == "critical"


class TestPerformanceDegradation:
    """TC: throughput drops >20% vs baseline."""

    def test_no_degradation(self):
        metrics = [_make_metric(s, "perf/throughput", 100.0) for s in range(30)]
        report = run_stability_diagnosis(metrics)
        alerts = [a for a in report.alerts if a.rule_name == "performance_degradation"]
        assert len(alerts) == 0

    def test_degradation_detected(self):
        metrics = [_make_metric(s, "perf/throughput", 100.0) for s in range(20)]
        metrics += [_make_metric(s, "perf/throughput", 50.0) for s in range(20, 25)]
        report = run_stability_diagnosis(metrics)
        alerts = [a for a in report.alerts if a.rule_name == "performance_degradation"]
        assert len(alerts) >= 1


class TestReportStructure:
    """TC: report has correct structure and metadata."""

    def test_empty_metrics(self):
        report = run_stability_diagnosis(metrics=[])
        assert report.total_alerts == 0
        assert len(report.rules_checked) > 0
        assert report.config_version == STABILITY_CONFIG_VERSION

    def test_phase_isolation(self):
        """Validation phase metrics should not mix with train."""
        train_metrics = [_make_metric(s, "perf/throughput", 100.0, "train") for s in range(25)]
        val_metrics = [_make_metric(s, "perf/throughput", 10.0, "validation") for s in range(25)]
        report = run_stability_diagnosis(metrics=train_metrics + val_metrics)
        perf_alerts = [a for a in report.alerts if a.rule_name == "performance_degradation"]
        # Train phase should not trigger (100 is stable)
        train_alerts = [a for a in perf_alerts if a.phase == "train"]
        assert len(train_alerts) == 0

    def test_alert_has_all_fields(self):
        metrics = [_make_metric(s, "actor/entropy", 2.0) for s in range(20)]
        metrics += [_make_metric(s, "actor/entropy", 0.1) for s in range(20, 30)]
        report = run_stability_diagnosis(metrics)
        assert len(report.alerts) > 0
        alert = report.alerts[0]
        assert alert.rule_name
        assert alert.severity in ("warning", "critical", "info")
        assert alert.metric_name
        assert alert.config_version == STABILITY_CONFIG_VERSION
        assert len(alert.investigation_hints) > 0
        assert len(alert.window_steps) > 0
        assert alert.description

    def test_config_versioning(self):
        config = StabilityConfig()
        assert config.config_version == STABILITY_CONFIG_VERSION
        d = config.to_dict()
        assert "config_version" in d
        assert "explore_collapse" in d
