"""Tests for REQ-12: Rule validity and migration evaluation framework."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_diagnostics.rule_eval.rule_card import (
    RuleCard, RuleRegistry, create_default_rules, InterventionRecord,
    MigrationResult,
)
from rl_diagnostics.rule_eval.experiment_matrix import (
    ExperimentMatrix, ExperimentConfig, create_default_matrix,
)
from rl_diagnostics.rule_eval.rule_validator import (
    validate_rule, compute_precision_recall, compute_seed_consistency,
    compute_intervention_effectiveness, LabeledWindow, SeedResult,
    ValidationReport, update_rule_card,
)
from rl_diagnostics.rule_eval.migration_report import (
    generate_migration_report, assess_migration, MigrationReport,
    MigrationValidationResult,
)


class TestRuleCard:
    def test_create_card(self):
        card = RuleCard(
            rule_name="test_rule",
            rule_version="0.1.0",
            applicable_algorithms=["GRPO"],
            trigger_condition="test condition",
        )
        assert card.rule_name == "test_rule"
        assert card.status == "experimental"

    def test_to_markdown(self):
        card = RuleCard(rule_name="test", trigger_condition="cond")
        md = card.to_markdown()
        assert "test" in md
        assert "experimental" in md

    def test_to_dict(self):
        card = RuleCard(rule_name="test")
        d = card.to_dict()
        assert d["rule_name"] == "test"
        assert "rule_version" in d


class TestRuleRegistry:
    def test_register_and_get(self):
        reg = RuleRegistry()
        card = RuleCard(rule_name="r1")
        reg.register(card)
        assert reg.get("r1") is card
        assert reg.get("nonexistent") is None

    def test_all_rules(self):
        reg = RuleRegistry()
        reg.register(RuleCard(rule_name="r1"))
        reg.register(RuleCard(rule_name="r2"))
        assert len(reg.all_rules()) == 2

    def test_validated_vs_experimental(self):
        reg = RuleRegistry()
        c1 = RuleCard(rule_name="r1", status="validated")
        c2 = RuleCard(rule_name="r2", status="experimental")
        reg.register(c1)
        reg.register(c2)
        assert len(reg.validated_rules()) == 1
        assert len(reg.experimental_rules()) == 1

    def test_default_rules(self):
        reg = create_default_rules()
        rules = reg.all_rules()
        assert len(rules) == 3
        names = {r.rule_name for r in rules}
        assert "low_information_group" in names
        assert "output_truncation" in names
        assert "kl_drift" in names

    def test_default_rules_have_content(self):
        reg = create_default_rules()
        for rule in reg.all_rules():
            assert rule.trigger_condition != ""
            assert len(rule.applicable_algorithms) > 0
            assert len(rule.typical_badcases) > 0
            assert len(rule.candidate_causes) > 0
            assert len(rule.suggested_interventions) > 0

    def test_to_json(self):
        reg = create_default_rules()
        j = reg.to_json()
        assert "low_information_group" in j


class TestExperimentMatrix:
    def test_create_matrix(self):
        matrix = create_default_matrix()
        assert matrix.discovery_model != ""
        assert len(matrix.discovery_datasets) == 2
        assert len(matrix.discovery_seeds) == 3
        assert len(matrix.rules) == 3

    def test_generate_runs(self):
        matrix = create_default_matrix()
        runs = matrix.generate_runs()
        assert len(runs) > 0
        # Discovery: 3 rules * 2 datasets * 4 interventions * 3 seeds = 72
        # Migration: 3 rules * 4 interventions * 3 seeds = 36
        # Total: 108
        assert len(runs) == 108

    def test_estimate_total(self):
        matrix = create_default_matrix()
        total = matrix.estimate_total_runs()
        assert total == 108

    def test_run_ids(self):
        matrix = create_default_matrix()
        runs = matrix.generate_runs()
        discovery = [r for r in runs if r.phase == "discovery"]
        migration = [r for r in runs if r.phase == "migration"]
        assert all(r.experiment_id.startswith("disc_") for r in discovery)
        assert all(r.experiment_id.startswith("migr_") for r in migration)

    def test_to_markdown(self):
        matrix = create_default_matrix()
        matrix.generate_runs()
        md = matrix.to_markdown()
        assert "Experiment Matrix" in md
        assert "Discovery" in md
        assert "Migration" in md


class TestPrecisionRecall:
    def test_perfect_rule(self):
        windows = [
            LabeledWindow(rule_fired=True, anomaly_present=True),
            LabeledWindow(rule_fired=False, anomaly_present=False),
        ]
        prec, rec, tp, fp, fn, tn = compute_precision_recall(windows)
        assert prec == 1.0
        assert rec == 1.0
        assert tp == 1
        assert fp == 0
        assert fn == 0

    def test_false_positive(self):
        windows = [
            LabeledWindow(rule_fired=True, anomaly_present=True),
            LabeledWindow(rule_fired=True, anomaly_present=False),
        ]
        prec, rec, tp, fp, fn, tn = compute_precision_recall(windows)
        assert prec == 0.5
        assert fp == 1

    def test_false_negative(self):
        windows = [
            LabeledWindow(rule_fired=False, anomaly_present=True),
        ]
        prec, rec, tp, fp, fn, tn = compute_precision_recall(windows)
        assert rec == 0.0
        assert fn == 1


class TestSeedConsistency:
    def test_all_consistent(self):
        seeds = [SeedResult(seed=0, target_metrics={"acc": 0.1}),
                 SeedResult(seed=1, target_metrics={"acc": 0.2}),
                 SeedResult(seed=2, target_metrics={"acc": 0.15})]
        consistency, same, total = compute_seed_consistency(seeds)
        assert consistency == 1.0
        assert total == 3

    def test_mixed(self):
        seeds = [SeedResult(seed=0, target_metrics={"acc": 0.1}),
                 SeedResult(seed=1, target_metrics={"acc": -0.1}),
                 SeedResult(seed=2, target_metrics={"acc": 0.15})]
        consistency, same, total = compute_seed_consistency(seeds)
        assert same == 2  # 2 improved, 1 didn't
        assert consistency == pytest.approx(2 / 3)

    def test_empty(self):
        consistency, same, total = compute_seed_consistency([])
        assert consistency == 0.0


class TestInterventionEffectiveness:
    def test_improvement(self):
        baseline = [SeedResult(seed=0, target_metrics={"acc": 0.5})]
        intervention = [SeedResult(seed=0, target_metrics={"acc": 0.7})]
        eff = compute_intervention_effectiveness(
            baseline, intervention, target_metric_name="acc"
        )
        assert eff["target_metric_improvement"] == pytest.approx(40.0)  # (0.7-0.5)/0.5 * 100

    def test_no_change(self):
        baseline = [SeedResult(seed=0, target_metrics={"acc": 0.5})]
        intervention = [SeedResult(seed=0, target_metrics={"acc": 0.5})]
        eff = compute_intervention_effectiveness(
            baseline, intervention, target_metric_name="acc"
        )
        assert eff["target_metric_improvement"] == 0.0


class TestValidateRule:
    def test_passing_rule(self):
        windows = [
            LabeledWindow(rule_fired=True, anomaly_present=True),
            LabeledWindow(rule_fired=True, anomaly_present=True),
            LabeledWindow(rule_fired=False, anomaly_present=False),
            LabeledWindow(rule_fired=False, anomaly_present=False),
        ]
        seeds = [
            SeedResult(seed=i, target_metrics={"metric": 0.1})
            for i in range(3)
        ]
        intervention_data = {
            "baseline_seeds": [SeedResult(seed=0, target_metrics={"acc": 0.5})],
            "intervention_seeds": [SeedResult(seed=0, target_metrics={"acc": 0.7})],
            "target_metric_name": "acc",
        }
        report = validate_rule("test_rule", windows, seeds, intervention_data)
        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.meets_precision is True
        assert report.meets_recall is True
        assert report.meets_seed_consistency is True
        assert report.meets_intervention is True
        assert report.overall_pass is True

    def test_failing_rule(self):
        windows = [
            LabeledWindow(rule_fired=True, anomaly_present=False),  # FP
            LabeledWindow(rule_fired=False, anomaly_present=True),  # FN
        ]
        report = validate_rule("test_rule", windows, [], None)
        assert report.precision == 0.0
        assert report.recall == 0.0
        assert report.overall_pass is False

    def test_update_rule_card(self):
        card = RuleCard(rule_name="test", status="experimental")
        report = ValidationReport(rule_name="test", overall_pass=True)
        updated = update_rule_card(card, report)
        assert updated.status == "validated"
        assert updated.validation_precision is not None


class TestMigrationReport:
    def test_assess_migration_no_data(self):
        disc_report = ValidationReport(rule_name="test", precision=0.9, recall=0.85)
        result = assess_migration("test", disc_report, [])
        assert result.notes != ""
        assert result.meets_criteria is False

    def test_assess_migration_consistent(self):
        disc_report = ValidationReport(
            rule_name="test", precision=0.9, recall=0.85,
            target_metric_improvement=30.0,
        )
        mig_seeds = [
            SeedResult(seed=0, rule_fired=True, target_metrics={"acc": 20}),
            SeedResult(seed=1, rule_fired=True, target_metrics={"acc": 25}),
            SeedResult(seed=2, rule_fired=True, target_metrics={"acc": 22}),
        ]
        result = assess_migration("test", disc_report, mig_seeds)
        assert result.direction_consistent is True
        assert result.seeds_consistent == 3
        assert result.meets_criteria is True

    def test_generate_full_report(self):
        reg = create_default_rules()
        discovery_reports = {
            "low_information_group": ValidationReport(
                rule_name="low_information_group",
                precision=0.85, recall=0.82, overall_pass=True,
            ),
        }
        migration_data = {
            "low_information_group": {
                "seed_results": [
                    SeedResult(seed=0, rule_fired=True),
                    SeedResult(seed=1, rule_fired=True),
                    SeedResult(seed=2, rule_fired=True),
                ],
            },
        }
        matrix = create_default_matrix()
        report = generate_migration_report(
            reg, discovery_reports, migration_data, experiment_matrix=matrix
        )
        assert len(report.rules_validated) == 1
        assert report.rules_validated[0].rule_name == "low_information_group"
        assert isinstance(report.summary, str)

    def test_report_to_markdown(self):
        report = MigrationReport(summary="Test summary")
        md = report.to_markdown()
        assert "Migration" in md
        assert "Test summary" in md

    def test_report_to_dict(self):
        report = MigrationReport()
        d = report.to_dict()
        assert "rules_validated" in d
        assert "overall_pass" in d
