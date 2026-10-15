"""REQ-12: Migration report generation.

Generates migration validation reports that assess whether diagnostic
rules discovered on one dataset/model transfer to unseen datasets and
larger models. This is the reporting framework — actual migration data
must come from real experiments.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from .rule_card import RuleCard, RuleRegistry, MigrationResult
from .rule_validator import ValidationReport, SeedResult
from .experiment_matrix import ExperimentMatrix


@dataclass
class MigrationValidationResult:
    """Result of migration validation for a single rule."""
    rule_name: str = ""
    discovery_precision: float = 0.0
    discovery_recall: float = 0.0
    migration_precision: float = 0.0
    migration_recall: float = 0.0
    direction_consistent: bool = False
    seeds_consistent: int = 0
    seeds_total: int = 0
    effect_retention: float = 0.0  # retained effect relative to discovery
    meets_criteria: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class MigrationReport:
    """Complete migration report across all rules."""
    config_version: str = "0.1.0-exp"
    rules_validated: list[MigrationValidationResult] = field(default_factory=list)
    overall_pass: bool = False
    summary: str = ""
    experiment_matrix: Optional[dict] = None

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# Migration Validation Report",
            f"",
            f"**Config Version:** {self.config_version}",
            f"**Overall:** {'PASS' if self.overall_pass else 'PENDING/FAIL'}",
            f"",
            f"## Per-Rule Results",
        ]
        for rv in self.rules_validated:
            lines.append(f"")
            lines.append(f"### {rv.rule_name}")
            lines.append(f"- Discovery: P={rv.discovery_precision:.3f}, R={rv.discovery_recall:.3f}")
            lines.append(f"- Migration: P={rv.migration_precision:.3f}, R={rv.migration_recall:.3f}")
            lines.append(f"- Direction consistent: {rv.direction_consistent}")
            lines.append(f"- Seeds: {rv.seeds_consistent}/{rv.seeds_total}")
            lines.append(f"- Effect retention: {rv.effect_retention:.1%}")
            lines.append(f"- Meets criteria: {'✓' if rv.meets_criteria else '✗'}")
            if rv.notes:
                lines.append(f"- Notes: {rv.notes}")

        lines.append(f"")
        lines.append(f"## Summary")
        lines.append(self.summary or "No summary available.")
        return "\n".join(lines)


def assess_migration(
    rule_name: str,
    discovery_report: ValidationReport,
    migration_seed_results: list[SeedResult],
    migration_windows: Optional[list] = None,
) -> MigrationValidationResult:
    """Assess migration of a rule from discovery to unseen dataset/model.

    Args:
        rule_name: Name of the rule.
        discovery_report: Validation report from discovery phase.
        migration_seed_results: Per-seed results from migration experiments.
        migration_windows: Optional labeled windows from migration phase.

    Returns:
        MigrationValidationResult with assessment.
    """
    result = MigrationValidationResult(
        rule_name=rule_name,
        discovery_precision=discovery_report.precision,
        discovery_recall=discovery_report.recall,
    )

    if not migration_seed_results:
        result.notes = "No migration seed results available"
        return result

    result.seeds_total = len(migration_seed_results)

    # Check direction consistency
    fired_count = sum(1 for s in migration_seed_results if s.rule_fired)
    result.direction_consistent = (
        (fired_count >= 2 and len(migration_seed_results) >= 3) or
        (fired_count == 0 and len(migration_seed_results) >= 3 and
         not discovery_report.meets_precision)
    )
    result.seeds_consistent = max(fired_count, len(migration_seed_results) - fired_count)

    # Migration precision/recall from windows if available
    if migration_windows:
        from .rule_validator import compute_precision_recall
        prec, rec, _, _, _, _ = compute_precision_recall(migration_windows)
        result.migration_precision = prec
        result.migration_recall = rec

    # Effect retention
    if discovery_report.target_metric_improvement > 0:
        # Compare average target metric improvement in migration vs discovery
        migration_improvements = [
            sum(s.target_metrics.values()) / max(len(s.target_metrics), 1)
            for s in migration_seed_results if s.target_metrics
        ]
        if migration_improvements:
            avg_migration = sum(migration_improvements) / len(migration_improvements)
            result.effect_retention = avg_migration / discovery_report.target_metric_improvement

    # Check criteria
    result.meets_criteria = (
        result.direction_consistent and
        result.seeds_consistent >= 2 and
        result.seeds_total >= 3 and
        result.seeds_consistent / max(result.seeds_total, 1) >= 2 / 3
    )

    return result


def generate_migration_report(
    registry: RuleRegistry,
    discovery_reports: dict[str, ValidationReport],
    migration_data: dict[str, dict],  # rule_name -> {seed_results, windows}
    experiment_matrix: Optional[ExperimentMatrix] = None,
    output_path: Optional[str] = None,
) -> MigrationReport:
    """Generate a complete migration report.

    Args:
        registry: Rule registry with all rule cards.
        discovery_reports: Dict of rule_name -> discovery ValidationReport.
        migration_data: Dict of rule_name -> {"seed_results": [...], "windows": [...]}.
        experiment_matrix: Optional experiment matrix for context.
        output_path: Optional path to save JSON report.

    Returns:
        MigrationReport object.
    """
    report = MigrationReport()

    if experiment_matrix:
        report.experiment_matrix = experiment_matrix.to_dict()

    all_pass = True
    for rule in registry.all_rules():
        disc_report = discovery_reports.get(rule.rule_name)
        if disc_report is None:
            continue

        mig_data = migration_data.get(rule.rule_name, {})
        mig_result = assess_migration(
            rule_name=rule.rule_name,
            discovery_report=disc_report,
            migration_seed_results=mig_data.get("seed_results", []),
            migration_windows=mig_data.get("windows"),
        )

        # Update rule card
        rule.migration_results.append(MigrationResult(
            dataset=experiment_matrix.migration_dataset if experiment_matrix else "",
            model=experiment_matrix.migration_model if experiment_matrix else "",
            seeds=[s.seed for s in mig_data.get("seed_results", [])],
            direction_consistent=mig_result.direction_consistent,
            effect_size=mig_result.effect_retention,
            effect_retention=mig_result.effect_retention,
            notes=mig_result.notes,
        ))

        if mig_result.meets_criteria:
            rule.status = "validated"
        else:
            all_pass = False

        report.rules_validated.append(mig_result)

    report.overall_pass = all_pass and len(report.rules_validated) > 0

    # Generate summary
    validated = sum(1 for r in report.rules_validated if r.meets_criteria)
    total = len(report.rules_validated)
    report.summary = (
        f"{validated}/{total} rules passed migration validation. "
        f"Overall: {'PASS' if report.overall_pass else 'PENDING/FAIL'}."
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    return report
