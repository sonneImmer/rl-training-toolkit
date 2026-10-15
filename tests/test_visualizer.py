"""Tests for REQ-04: Rollout visualization."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_diagnostics.analyzers.group_signal import StepAggregation
from rl_diagnostics.analyzers.rollout_quality import RolloutQualityResult
from rl_diagnostics.analyzers.difficulty_analyzer import DifficultyReport, DifficultyStats
from rl_diagnostics.reporters.visualizer import (
    chart_step_signal, chart_epoch_comparison,
    chart_rollout_quality, chart_difficulty_stratification,
    SVGChart, ChartTheme,
)


def _make_step_agg(step, epoch=None, **kwargs):
    defaults = dict(
        total_groups=10, analyzable_groups=10,
        effective_groups=5, constant_score_groups=3,
        near_constant_groups=1, insufficient_groups=1,
        unresolved_groups=0, total_rollouts=50, usable_rollouts=45,
        rollout_usable_ratio=0.9, effective_group_ratio=0.5,
        low_information_group_ratio=0.4, constant_score_group_ratio=0.3,
        near_constant_group_ratio=0.1, mean_informative_pair_ratio=0.6,
    )
    defaults.update(kwargs)
    return StepAggregation(step=step, epoch=epoch, **defaults)


class TestStepSignalChart:
    """TC-04-01: step chart covers all parsed steps."""

    def test_renders_svg(self):
        aggs = [_make_step_agg(s, epoch=s // 50) for s in range(1, 11)]
        svg = chart_step_signal(aggs)
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert "Per-Step Learning Signal" in svg

    def test_empty_data(self):
        svg = chart_step_signal([])
        assert "No step data" in svg

    def test_all_steps_present(self):
        """TC-04-01: all steps should be covered."""
        aggs = [_make_step_agg(s) for s in range(1, 6)]
        svg = chart_step_signal(aggs)
        for s in range(1, 6):
            assert str(s) in svg

    def test_high_constant_ratio(self):
        """TC-04-02: 80% constant-score should be visible."""
        aggs = [_make_step_agg(1, effective_groups=1, constant_score_groups=8,
                               effective_group_ratio=0.1,
                               low_information_group_ratio=0.8,
                               constant_score_group_ratio=0.8)]
        svg = chart_step_signal(aggs)
        assert "<svg" in svg
        # Red color for constant should be present
        assert "#f44336" in svg


class TestEpochComparison:
    """TC: epoch comparison chart."""

    def test_renders(self):
        aggs = [_make_step_agg(s, epoch=s // 50) for s in range(1, 101)]
        svg = chart_epoch_comparison(aggs)
        assert "<svg" in svg
        assert "Epoch Comparison" in svg

    def test_no_epoch_data(self):
        aggs = [_make_step_agg(1, epoch=None)]
        svg = chart_epoch_comparison(aggs)
        assert "No epoch data" in svg


class TestRolloutQualityChart:
    """TC: rollout quality distribution chart."""

    def test_renders(self):
        results = [
            RolloutQualityResult("exp", 1, "g1", 0, "usable"),
            RolloutQualityResult("exp", 1, "g1", 1, "usable"),
            RolloutQualityResult("exp", 1, "g2", 0, "invalid_empty"),
            RolloutQualityResult("exp", 1, "g2", 1, "invalid_reward"),
        ]
        svg = chart_rollout_quality(results)
        assert "<svg" in svg
        assert "Rollout Data Quality" in svg
        assert "Usable" in svg
        assert "Empty" in svg

    def test_empty(self):
        svg = chart_rollout_quality([])
        assert "No quality data" in svg


class TestDifficultyChart:
    """TC-04-03/04: difficulty chart conditional rendering."""

    def test_no_difficulty(self):
        """TC-04-03: should show 'not provided' message."""
        report = DifficultyReport(has_difficulty=False)
        svg = chart_difficulty_stratification(report)
        assert "Difficulty field not provided" in svg
        assert "No automatic difficulty inference" in svg

    def test_with_difficulty(self):
        """TC-04-04: should show buckets by original labels."""
        report = DifficultyReport(has_difficulty=True, total_groups=20)
        report.buckets = {
            "easy": DifficultyStats(difficulty="easy", group_count=10,
                                    effective_group_ratio=0.7,
                                    constant_score_group_ratio=0.2,
                                    near_constant_group_ratio=0.1),
            "hard": DifficultyStats(difficulty="hard", group_count=10,
                                    effective_group_ratio=0.3,
                                    constant_score_group_ratio=0.5,
                                    near_constant_group_ratio=0.2),
        }
        svg = chart_difficulty_stratification(report)
        assert "<svg" in svg
        assert "easy" in svg
        assert "hard" in svg

    def test_empty_buckets(self):
        report = DifficultyReport(has_difficulty=True)
        svg = chart_difficulty_stratification(report)
        assert "No difficulty buckets" in svg


class TestSVGChart:
    """Test SVG primitive builder."""

    def test_basic_elements(self):
        chart = SVGChart(200, 100)
        chart.rect(10, 10, 50, 50, fill="#ff0000")
        chart.text(20, 30, "Hello")
        chart.line(0, 0, 100, 100, stroke="#000")
        svg = chart.render()
        assert "<rect" in svg
        assert "<text" in svg
        assert "<line" in svg
        assert 'width="200"' in svg
        assert 'height="100"' in svg

    def test_escaping(self):
        chart = SVGChart(200, 100)
        chart.text(10, 10, "<script>alert(1)</script>")
        svg = chart.render()
        assert "<script>" not in svg
        assert "&lt;script&gt;" in svg


# ── Tests for intermediate-process charts ────────────────────────

from rl_diagnostics.analyzers.group_signal import GroupSignalResult
from rl_diagnostics.reporters.visualizer import (
    chart_group_reward_distribution,
    chart_rollout_quality_timeline,
    chart_group_signal_heatmap,
    chart_invalid_prompt_cards,
    export_all_svgs,
)


def _make_group_signal(step=1, group_id="g1", signal_type="effective_group",
                       rewards=None, ipr=0.6, **kwargs):
    defaults = dict(
        experiment_id="exp_test",
        step=step,
        group_id=group_id,
        signal_type=signal_type,
        reason="test",
        rollout_count=5,
        usable_count=5,
        unique_reward_count=2,
        reward_range=2.0,
        reward_std=0.98,
        informative_pair_ratio=ipr,
        low_information=False,
        diagnostic_reason="test reason",
    )
    defaults.update(kwargs)
    gs = GroupSignalResult(**defaults)
    gs.rewards = rewards if rewards is not None else [1.0, -1.0, 1.0, -1.0, 1.0]
    return gs


def _make_quality_result(step=1, group_id="g1", rollout_id=0,
                         status="usable", reason=""):
    return RolloutQualityResult(
        experiment_id="exp_test",
        step=step,
        group_id=group_id,
        rollout_id=rollout_id,
        status=status,
        reason=reason,
    )


class TestGroupRewardDistribution:
    """Tests for Chart 5: Group Reward Distribution."""

    def test_renders_svg(self):
        groups = [_make_group_signal(step=1, group_id=f"g{i}",
                                     rewards=[1.0, -1.0, 0.5, -0.5, 1.0])
                  for i in range(3)]
        svg = chart_group_reward_distribution(groups)
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert "Group Reward Distribution" in svg

    def test_empty_data(self):
        svg = chart_group_reward_distribution([])
        assert "No group data" in svg

    def test_signal_type_colors(self):
        """Groups with different signal types should produce SVG with
        the corresponding theme colors."""
        groups = [
            _make_group_signal(step=1, group_id="eff", signal_type="effective_group"),
            _make_group_signal(step=1, group_id="con", signal_type="constant_score",
                               rewards=[1.0, 1.0, 1.0, 1.0, 1.0]),
        ]
        svg = chart_group_reward_distribution(groups)
        assert "#4caf50" in svg  # effective color
        assert "#f44336" in svg  # constant color

    def test_all_groups_labeled(self):
        groups = [_make_group_signal(step=s, group_id=f"g{s}")
                  for s in range(1, 6)]
        svg = chart_group_reward_distribution(groups)
        for s in range(1, 6):
            assert f"s{s}" in svg

    def test_uniform_rewards(self):
        """All rewards identical should still render without error."""
        groups = [_make_group_signal(step=1, group_id="g1",
                                     signal_type="constant_score",
                                     rewards=[0.5, 0.5, 0.5])]
        svg = chart_group_reward_distribution(groups)
        assert "<svg" in svg


class TestRolloutQualityTimeline:
    """Tests for Chart 6: Rollout Quality Timeline."""

    def test_renders_svg(self):
        results = [
            _make_quality_result(step=1, rollout_id=i, status="usable")
            for i in range(5)
        ] + [
            _make_quality_result(step=2, rollout_id=i,
                                 status="invalid_reward" if i == 0 else "usable")
            for i in range(5)
        ]
        svg = chart_rollout_quality_timeline(results)
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert "Rollout Quality Timeline" in svg

    def test_empty_data(self):
        svg = chart_rollout_quality_timeline([])
        assert "No quality data" in svg

    def test_multiple_steps_shown(self):
        results = [
            _make_quality_result(step=s, rollout_id=i, status="usable")
            for s in range(1, 4) for i in range(5)
        ]
        svg = chart_rollout_quality_timeline(results)
        for s in range(1, 4):
            assert f"Step {s}" in svg

    def test_degraded_step_visible(self):
        """A step with many invalid rollouts should show non-green segments."""
        results = [
            _make_quality_result(step=1, rollout_id=i, status="usable")
            for i in range(5)
        ] + [
            _make_quality_result(step=2, rollout_id=i,
                                 status="invalid_empty")
            for i in range(3)
        ] + [
            _make_quality_result(step=2, rollout_id=i, status="usable")
            for i in range(3, 5)
        ]
        svg = chart_rollout_quality_timeline(results)
        assert "#f44336" in svg  # quality_empty color

    def test_only_invalid(self):
        results = [_make_quality_result(step=1, status="invalid_format")]
        svg = chart_rollout_quality_timeline(results)
        assert "<svg" in svg


class TestGroupSignalHeatmap:
    """Tests for Chart 7: Group Signal Heatmap."""

    def test_renders_svg(self):
        groups = [
            _make_group_signal(step=1, group_id=f"g{i}",
                               signal_type=st)
            for i, st in enumerate(["effective_group", "constant_score",
                                    "insufficient_group"])
        ]
        svg = chart_group_signal_heatmap(groups)
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert "Group Signal Heatmap" in svg

    def test_empty_data(self):
        svg = chart_group_signal_heatmap([])
        assert "No group signal data" in svg

    def test_step_labels(self):
        groups = [_make_group_signal(step=s, group_id=f"g{s}")
                  for s in range(1, 5)]
        svg = chart_group_signal_heatmap(groups)
        for s in range(1, 5):
            assert f"Step {s}" in svg

    def test_ipr_in_cells(self):
        """When cells are large enough, IPR values should be rendered."""
        groups = [_make_group_signal(step=1, group_id="g1", ipr=0.75)]
        svg = chart_group_signal_heatmap(groups, width=900, height=400)
        assert "0.75" in svg

    def test_legend_colors(self):
        groups = [_make_group_signal(step=1, signal_type="effective_group")]
        svg = chart_group_signal_heatmap(groups)
        assert "#4caf50" in svg  # effective
        assert "#f44336" in svg  # constant


class TestInvalidPromptCards:
    """Tests for Chart 8: Abnormal Prompt Cards."""

    def test_renders_svg(self):
        prompts = [
            {"prompt": "What is 2+2?", "diagnostic_type": "constant_score",
             "diagnostic_reason": "All rewards identical",
             "rewards": [1.0, 1.0, 1.0], "triggered_steps": [1]},
        ]
        svg = chart_invalid_prompt_cards(prompts)
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert "Abnormal Prompt Cards" in svg

    def test_empty_data(self):
        svg = chart_invalid_prompt_cards([])
        assert "No abnormal prompts" in svg

    def test_multiple_cards(self):
        prompts = [
            {"prompt": f"Prompt {i}", "diagnostic_type": "insufficient_group",
             "diagnostic_reason": "Too few rollouts",
             "rewards": [1.0], "triggered_steps": [i]}
            for i in range(5)
        ]
        svg = chart_invalid_prompt_cards(prompts)
        assert "<svg" in svg
        # Should mention Insufficient Group badge
        assert "Insufficient" in svg

    def test_long_prompt_truncated(self):
        long_prompt = "A" * 200
        prompts = [
            {"prompt": long_prompt, "diagnostic_type": "constant_score",
             "diagnostic_reason": "test", "rewards": [1.0],
             "triggered_steps": [1]},
        ]
        svg = chart_invalid_prompt_cards(prompts)
        assert "..." in svg

    def test_reward_bar_rendered(self):
        prompts = [
            {"prompt": "test", "diagnostic_type": "constant_score",
             "diagnostic_reason": "test", "rewards": [1.0, -1.0, 0.5],
             "triggered_steps": [1]},
        ]
        svg = chart_invalid_prompt_cards(prompts)
        assert "Rewards:" in svg


class TestExportAllSvgs:
    """Tests for standalone SVG export."""

    def test_export_to_dir(self, tmp_path):
        step_aggs = [_make_step_agg(s) for s in range(1, 4)]
        quality_results = [
            _make_quality_result(step=1, rollout_id=i) for i in range(5)
        ]
        group_signals = [_make_group_signal(step=1, group_id=f"g{i}")
                         for i in range(3)]
        diff_report = DifficultyReport(has_difficulty=False)
        invalid_prompts = [
            {"prompt": "test", "diagnostic_type": "constant_score",
             "diagnostic_reason": "test", "rewards": [1.0],
             "triggered_steps": [1]},
        ]
        written = export_all_svgs(
            output_dir=str(tmp_path),
            step_aggs=step_aggs,
            quality_results=quality_results,
            group_signals=group_signals,
            diff_report=diff_report,
            invalid_prompts=invalid_prompts,
        )
        assert len(written) >= 7
        for path in written:
            assert os.path.isfile(path)
            with open(path) as f:
                content = f.read()
            assert content.startswith("<svg")
            assert content.endswith("</svg>")

    def test_export_empty_data(self, tmp_path):
        """Should not crash when no data is provided."""
        written = export_all_svgs(output_dir=str(tmp_path))
        # diff_report is None, so difficulty chart is skipped
        assert len(written) == 0

    def test_export_only_step_aggs(self, tmp_path):
        step_aggs = [_make_step_agg(1)]
        written = export_all_svgs(
            output_dir=str(tmp_path),
            step_aggs=step_aggs,
        )
        assert len(written) == 2  # step_signal + epoch_comparison
        assert any("step_signal" in p for p in written)
        assert any("epoch" in p for p in written)
