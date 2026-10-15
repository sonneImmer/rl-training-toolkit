"""Tests for REQ-03 Layer 2: Group learning signal diagnosis."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_diagnostics.schemas.data_models import GroupRecord, SCHEMA_VERSION
from rl_diagnostics.analyzers.rollout_quality import RolloutQualityResult
from rl_diagnostics.analyzers.group_signal import (
    diagnose_group_signal, aggregate_step, GroupSignalConfig,
)


def _make_group(group_id="g1", step=1, rewards=None, rollout_count=5,
                group_status="resolved", prompt="test") -> GroupRecord:
    if rewards is None:
        rewards = [1.0, -1.0, 1.0, -1.0, 1.0]
    return GroupRecord(
        schema_version=SCHEMA_VERSION,
        experiment_id="test", step=step, epoch=0, generation_attempt=None,
        group_id=group_id, prompt_id="pid", prompt=prompt,
        source_dataset="math", difficulty=None, ground_truth="42",
        rollout_count=rollout_count, rewards=rewards,
        responses=[f"resp{i}" for i in range(rollout_count)],
        rollout_ids=list(range(rollout_count)),
        group_status=group_status,
    )


def _make_quality_all_usable(group_id="g1", step=1, count=5):
    return [RolloutQualityResult(
        experiment_id="test", step=step, group_id=group_id,
        rollout_id=i, status="usable"
    ) for i in range(count)]


class TestConstantScore:
    def test_all_same_positive(self):
        """TC-03-01: reward [0.7,0.7,0.7,0.7,0.7], no correctness -> constant_score, not all-correct"""
        g = _make_group(rewards=[0.7, 0.7, 0.7, 0.7, 0.7])
        qs = _make_quality_all_usable()
        result = diagnose_group_signal(g, qs)
        assert result.signal_type == "constant_score"
        assert result.correctness_type is None  # no correctness labels
        assert result.low_information is True

    def test_all_same_negative(self):
        """TC-03-02: reward [-2,-2,-2,-2,-2], no correctness -> constant_score, not all-wrong"""
        g = _make_group(rewards=[-2.0, -2.0, -2.0, -2.0, -2.0])
        qs = _make_quality_all_usable()
        result = diagnose_group_signal(g, qs)
        assert result.signal_type == "constant_score"
        assert result.correctness_type is None
        assert result.low_information is True


class TestEffectiveGroup:
    def test_binary_distinct(self):
        """TC-03-03: reward [0.2,0.5,0.2,0.8,0.5] -> effective_group"""
        g = _make_group(rewards=[0.2, 0.5, 0.2, 0.8, 0.5])
        qs = _make_quality_all_usable()
        result = diagnose_group_signal(g, qs)
        assert result.signal_type == "effective_group"
        assert result.informative_pair_ratio > 0

    def test_clear_binary(self):
        """reward [1,0,1,0,1] -> effective"""
        g = _make_group(rewards=[1.0, 0.0, 1.0, 0.0, 1.0])
        qs = _make_quality_all_usable()
        result = diagnose_group_signal(g, qs)
        assert result.signal_type == "effective_group"


class TestNearConstant:
    def test_small_differences(self):
        """TC-03-04: continuous reward differences below threshold -> near_constant_score"""
        g = _make_group(rewards=[0.50, 0.51, 0.49, 0.50, 0.50])
        qs = _make_quality_all_usable()
        config = GroupSignalConfig(
            abs_epsilon=1e-6, rel_epsilon=1e-4,
            near_constant_abs_threshold=0.02,
            near_constant_rel_threshold=0.05,
        )
        result = diagnose_group_signal(g, qs, config)
        assert result.signal_type in ("near_constant_score", "constant_score")
        assert result.low_information is True


class TestInsufficientGroup:
    def test_missing_rewards(self):
        """TC-03-05: 5 rollouts, 2 missing reward -> insufficient or low usable"""
        g = _make_group(rewards=[1.0, 1.0, None, None, None], rollout_count=5)
        qs = [
            RolloutQualityResult(experiment_id="test", step=1, group_id="g1", rollout_id=0, status="usable"),
            RolloutQualityResult(experiment_id="test", step=1, group_id="g1", rollout_id=1, status="usable"),
            RolloutQualityResult(experiment_id="test", step=1, group_id="g1", rollout_id=2, status="invalid_reward"),
            RolloutQualityResult(experiment_id="test", step=1, group_id="g1", rollout_id=3, status="invalid_reward"),
            RolloutQualityResult(experiment_id="test", step=1, group_id="g1", rollout_id=4, status="invalid_reward"),
        ]
        config = GroupSignalConfig(min_usable_rollouts=3)
        result = diagnose_group_signal(g, qs, config)
        assert result.signal_type == "insufficient_group"
        assert result.usable_count == 2
        assert result.rollout_count == 5

    def test_too_few_rollouts(self):
        g = _make_group(rewards=[1.0], rollout_count=1)
        qs = [RolloutQualityResult(experiment_id="test", step=1, group_id="g1", rollout_id=0, status="usable")]
        result = diagnose_group_signal(g, qs)
        assert result.signal_type == "insufficient_group"


class TestCorrectnessSeparation:
    def test_constant_with_mixed_correctness(self):
        """TC-03-07: reward all same, but eval labels have mix -> low info, keep correctness evidence"""
        g = _make_group(rewards=[1.0, 1.0, 1.0, 1.0, 1.0])
        qs = _make_quality_all_usable()
        labels = [True, False, True, True, False]
        result = diagnose_group_signal(g, qs, correctness_labels=labels)
        assert result.signal_type == "constant_score"
        assert result.low_information is True
        assert result.correctness_type == "mixed_correctness"

    def test_constant_all_correct(self):
        g = _make_group(rewards=[1.0, 1.0, 1.0])
        qs = _make_quality_all_usable(count=3)
        labels = [True, True, True]
        result = diagnose_group_signal(g, qs, correctness_labels=labels)
        assert result.signal_type == "constant_score"
        assert result.correctness_type == "all_correct"

    def test_no_correctness_labels(self):
        g = _make_group(rewards=[1.0, -1.0, 1.0])
        qs = _make_quality_all_usable(count=3)
        result = diagnose_group_signal(g, qs)
        assert result.correctness_type is None


class TestGroupUnresolved:
    def test_unresolved_group(self):
        g = _make_group(group_status="group_unresolved")
        qs = _make_quality_all_usable()
        result = diagnose_group_signal(g, qs)
        assert result.signal_type == "group_unresolved"


class TestMetrics:
    def test_unique_reward_count(self):
        g = _make_group(rewards=[1.0, 0.5, 1.0, 0.0, 1.0])
        qs = _make_quality_all_usable()
        result = diagnose_group_signal(g, qs)
        assert result.unique_reward_count == 3  # 1.0, 0.5, 0.0

    def test_reward_range(self):
        g = _make_group(rewards=[1.0, -1.0, 0.5, 0.0, 0.3])
        qs = _make_quality_all_usable()
        result = diagnose_group_signal(g, qs)
        assert result.reward_range == pytest.approx(2.0)

    def test_informative_pair_ratio(self):
        g = _make_group(rewards=[1.0, 1.0, -1.0, -1.0, 1.0])
        qs = _make_quality_all_usable()
        result = diagnose_group_signal(g, qs)
        # 5 rollouts -> 10 pairs; distinguishable pairs: (0,2),(0,3),(1,2),(1,3),(4,2),(4,3) = 6
        assert result.informative_pair_ratio == pytest.approx(0.6)


class TestStepAggregation:
    def test_aggregate_basic(self):
        groups_signals = []
        quality = []
        for gid, rewards in [("g1", [1.0, -1.0, 1.0, -1.0, 1.0]),
                              ("g2", [1.0, 1.0, 1.0, 1.0, 1.0]),
                              ("g3", [0.5, 0.0, 1.0, 0.5, 0.0])]:
            g = _make_group(group_id=gid, rewards=rewards, rollout_count=len(rewards))
            qs = _make_quality_all_usable(group_id=gid, count=len(rewards))
            gs = diagnose_group_signal(g, qs)
            groups_signals.append(gs)
            quality.extend(qs)

        agg = aggregate_step(1, 0, groups_signals, quality)
        assert agg.total_groups == 3
        assert agg.effective_groups == 2  # g1 and g3
        assert agg.constant_score_groups == 1  # g2
        assert agg.effective_group_ratio == pytest.approx(2/3)
        assert agg.low_information_group_ratio == pytest.approx(1/3)

    def test_usable_ratio(self):
        quality = [
            RolloutQualityResult(experiment_id="t", step=1, group_id="g", rollout_id=0, status="usable"),
            RolloutQualityResult(experiment_id="t", step=1, group_id="g", rollout_id=1, status="usable"),
            RolloutQualityResult(experiment_id="t", step=1, group_id="g", rollout_id=2, status="invalid_empty"),
            RolloutQualityResult(experiment_id="t", step=1, group_id="g", rollout_id=3, status="invalid_reward"),
        ]
        agg = aggregate_step(1, 0, [], quality)
        assert agg.total_rollouts == 4
        assert agg.usable_rollouts == 2
        assert agg.rollout_usable_ratio == pytest.approx(0.5)


class TestConfigurableThresholds:
    def test_custom_abs_epsilon(self):
        """Large abs_epsilon makes small differences invisible."""
        g = _make_group(rewards=[1.0, 1.001, 0.999])
        qs = _make_quality_all_usable(count=3)
        config = GroupSignalConfig(abs_epsilon=0.01, rel_epsilon=0.0)
        result = diagnose_group_signal(g, qs, config)
        assert result.signal_type == "constant_score"

    def test_custom_min_effective_ratio(self):
        """High min_effective_ratio makes borderline groups near_constant."""
        g = _make_group(rewards=[1.0, 1.0, 1.0, 1.0, 0.0])
        qs = _make_quality_all_usable()
        config = GroupSignalConfig(min_effective_ratio=0.8)
        result = diagnose_group_signal(g, qs, config)
        # 4 same + 1 different: informative pairs = 4 out of 10 = 0.4 < 0.8
        assert result.signal_type == "near_constant_score"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
