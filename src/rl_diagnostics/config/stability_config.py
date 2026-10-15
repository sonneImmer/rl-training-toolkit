"""REQ-07: Configuration for training stability diagnosis rules.

All thresholds are first-round experimental starting points.
They must be calibrated via REQ-12 before being treated as production rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

STABILITY_CONFIG_VERSION = "0.1.0-exp"


@dataclass
class StabilityRuleConfig:
    """Configuration for a single stability rule."""
    enabled: bool = True
    window_size: int = 10
    baseline_window: int = 20
    threshold: float = 0.0
    min_windows: int = 2
    phase_isolated: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class StabilityConfig:
    """Top-level configuration for all stability rules."""
    config_version: str = STABILITY_CONFIG_VERSION

    # Explore collapse: actor/entropy median drops >50% vs baseline
    explore_collapse: StabilityRuleConfig = field(
        default_factory=lambda: StabilityRuleConfig(
            threshold=0.50, window_size=10, baseline_window=20,
            extra={"metric_names": ["actor/entropy", "actor/entropy_coeff"]},
        )
    )

    # KL drift: |actor/ppo_kl| or |actor/kl_loss| median > 3x baseline for 2 windows
    kl_drift: StabilityRuleConfig = field(
        default_factory=lambda: StabilityRuleConfig(
            threshold=3.0, window_size=10, baseline_window=20, min_windows=2,
            extra={"metric_names": ["actor/ppo_kl", "actor/kl_loss"]},
        )
    )

    # Clip saturation: actor/pg_clipfrac consecutive windows > 0.20
    clip_saturation: StabilityRuleConfig = field(
        default_factory=lambda: StabilityRuleConfig(
            threshold=0.20, window_size=10, baseline_window=20, min_windows=3,
            extra={"metric_names": ["actor/pg_clipfrac"]},
        )
    )

    # Gradient spike: actor/grad_norm > rolling_median + 5*MAD
    gradient_spike: StabilityRuleConfig = field(
        default_factory=lambda: StabilityRuleConfig(
            threshold=5.0, window_size=1, baseline_window=30,
            extra={"metric_names": ["actor/grad_norm"], "mad_multiplier": 5.0},
        )
    )

    # Output truncation: response_length/clip_ratio > 0.20 or rises 10pp vs baseline
    output_truncation: StabilityRuleConfig = field(
        default_factory=lambda: StabilityRuleConfig(
            threshold=0.20, window_size=10, baseline_window=20,
            extra={
                "metric_names": ["response_length/clip_ratio", "rollout/response_length/clip_ratio"],
                "relative_rise_pp": 10.0,
            },
        )
    )

    # Reward signal degradation: low_information_group_ratio > 0.80
    # or informative_pair_ratio persistently declining
    reward_signal_degradation: StabilityRuleConfig = field(
        default_factory=lambda: StabilityRuleConfig(
            threshold=0.80, window_size=5, baseline_window=10, min_windows=2,
            extra={
                "low_info_metric": "low_information_group_ratio",
                "ipr_metric": "mean_informative_pair_ratio",
                "ipr_decline_pct": 0.30,
            },
        )
    )

    # Performance degradation: throughput drops >20% vs same-phase baseline
    performance_degradation: StabilityRuleConfig = field(
        default_factory=lambda: StabilityRuleConfig(
            threshold=0.20, window_size=5, baseline_window=20,
            extra={
                "metric_names": ["perf/throughput", "timing_s/total", "timing_s/generate",
                                 "timing_s/reward_compute", "timing_s/actor_update"],
                "mode": "relative_drop",
            },
        )
    )

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    def get_rule_configs(self) -> dict[str, StabilityRuleConfig]:
        return {
            "explore_collapse": self.explore_collapse,
            "kl_drift": self.kl_drift,
            "clip_saturation": self.clip_saturation,
            "gradient_spike": self.gradient_spike,
            "output_truncation": self.output_truncation,
            "reward_signal_degradation": self.reward_signal_degradation,
            "performance_degradation": self.performance_degradation,
        }
