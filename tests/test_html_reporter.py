"""Tests for REQ-11: Local HTML diagnostic report."""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_diagnostics.schemas.data_models import ExperimentRecord, ParseReport
from rl_diagnostics.analyzers.group_signal import StepAggregation
from rl_diagnostics.analyzers.rollout_quality import RolloutQualityResult
from rl_diagnostics.analyzers.stability_analyzer import StabilityReport, StabilityAlert
from rl_diagnostics.analyzers.difficulty_analyzer import DifficultyReport
from rl_diagnostics.reporters.html_reporter import generate_html_report


def _make_step_agg(step, epoch=None):
    return StepAggregation(
        step=step, epoch=epoch or 0,
        total_groups=10, analyzable_groups=10,
        effective_groups=5, constant_score_groups=3,
        near_constant_groups=1, insufficient_groups=1,
        effective_group_ratio=0.5, low_information_group_ratio=0.4,
        constant_score_group_ratio=0.3, near_constant_group_ratio=0.1,
        mean_informative_pair_ratio=0.6,
    )


class TestHTMLReport:
    """TC: report generates valid self-contained HTML."""

    def test_basic_report(self, tmp_path):
        html_path = str(tmp_path / "report.html")
        result = generate_html_report(
            experiment=ExperimentRecord(experiment_id="test_exp", model="test-model",
                                        algorithm="grpo"),
            step_aggs=[_make_step_agg(s, s // 50) for s in range(1, 11)],
            quality_results=[RolloutQualityResult("exp", 1, "g1", 0, "usable")],
            summary={"total_rollouts": 32, "total_groups": 10},
            output_path=html_path,
        )
        assert result == html_path
        assert os.path.isfile(html_path)
        with open(html_path, encoding="utf-8") as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content
        assert "</html>" in content
        assert "test_exp" in content
        assert "test-model" in content

    def test_offline_no_external_deps(self, tmp_path):
        """TC: no external JS/CSS references."""
        html_path = str(tmp_path / "report.html")
        generate_html_report(
            experiment=ExperimentRecord(experiment_id="test"),
            output_path=html_path,
        )
        with open(html_path, encoding="utf-8") as f:
            content = f.read()
        # SVG namespace uses http://www.w3.org - that is allowed
        # We check for external resource references, not namespace URIs
        assert "<script src" not in content
        assert "<link rel=" not in content
        assert 'src="http' not in content
        assert 'href="http' not in content

    def test_embedded_svg(self, tmp_path):
        """TC: charts are embedded SVG, not external images."""
        html_path = str(tmp_path / "report.html")
        generate_html_report(
            step_aggs=[_make_step_agg(1)],
            output_path=html_path,
        )
        with open(html_path, encoding="utf-8") as f:
            content = f.read()
        assert "<svg" in content
        assert "<img" not in content

    def test_stability_alerts(self, tmp_path):
        """TC: stability alerts rendered as cards."""
        html_path = str(tmp_path / "report.html")
        report = StabilityReport(
            config_version="0.1.0-exp",
            total_alerts=1, critical_count=1,
            alerts=[StabilityAlert(
                rule_name="explore_collapse", severity="critical",
                metric_name="actor/entropy",
                trigger_value=0.5, baseline_value=2.0, threshold=0.5,
                window_steps=[25, 26, 27],
                description="Entropy dropped 75%",
                investigation_hints=["Check entropy coefficient"],
            )],
            rules_checked=["explore_collapse"],
        )
        generate_html_report(
            stability_report=report,
            output_path=html_path,
        )
        with open(html_path, encoding="utf-8") as f:
            content = f.read()
        assert "explore_collapse" in content
        assert "critical" in content
        assert "Entropy dropped" in content
        assert "Check entropy coefficient" in content

    def test_difficulty_missing(self, tmp_path):
        """TC: difficulty missing shows degraded message."""
        html_path = str(tmp_path / "report.html")
        generate_html_report(
            diff_report=DifficultyReport(has_difficulty=False),
            output_path=html_path,
        )
        with open(html_path, encoding="utf-8") as f:
            content = f.read()
        assert "Difficulty field not provided" in content or "not provided" in content

    def test_invalid_prompts_entry(self, tmp_path):
        """TC: invalid prompts file is linked."""
        prompts_path = str(tmp_path / "invalid_prompts.jsonl")
        with open(prompts_path, "w") as f:
            for i in range(3):
                f.write(json.dumps({
                    "step": i, "group_id": f"g{i}",
                    "prompt": f"Prompt {i}",
                    "triggered_reasons": ["constant_score"],
                }) + "\n")

        html_path = str(tmp_path / "report.html")
        generate_html_report(
            output_path=html_path,
            invalid_prompts_path=prompts_path,
        )
        with open(html_path, encoding="utf-8") as f:
            content = f.read()
        assert "invalid_prompts.jsonl" in content
        assert "constant_score" in content

    def test_empty_report(self, tmp_path):
        """TC: report with no data still generates valid HTML."""
        html_path = str(tmp_path / "report.html")
        generate_html_report(output_path=html_path)
        assert os.path.isfile(html_path)
        with open(html_path, encoding="utf-8") as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content
        assert "Overview" in content

    def test_unevaluated_badges(self, tmp_path):
        """TC: unevaluated REQs shown as badges."""
        html_path = str(tmp_path / "report.html")
        generate_html_report(output_path=html_path)
        with open(html_path, encoding="utf-8") as f:
            content = f.read()
        assert "REQ-08" in content
        assert "REQ-12" in content
        assert "badge-skip" in content

    def test_nav_links(self, tmp_path):
        """TC: navigation links to sections."""
        html_path = str(tmp_path / "report.html")
        generate_html_report(output_path=html_path)
        with open(html_path, encoding="utf-8") as f:
            content = f.read()
        assert 'href="#overview"' in content
        assert 'href="#stability"' in content
        assert 'href="#signal"' in content
