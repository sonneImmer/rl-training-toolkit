"""REQ-12: Rule validation statistics.

Computes precision, recall, seed consistency, and intervention
effectiveness from experiment results. This is the statistical
framework — it does not fabricate experiment data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .rule_card import RuleCard, InterventionRecord, MigrationResult


@dataclass
class LabeledWindow:
    """A labeled time window for rule validation.

    Represents a segment of training where an anomaly was manually
    labeled as present (positive) or absent (negative).
    """
    step_start: int = 0
    step_end: int = 0
    rule_fired: bool = False  # did the diagnostic rule trigger in this window?
    anomaly_present: bool = False  # was an anomaly actually observed?
    notes: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class SeedResult:
    """Result from a single seed run."""
    seed: int = 0
    intervention: str = "baseline"
    target_metrics: dict[str, float] = field(default_factory=dict)
    non_target_metrics: dict[str, float] = field(default_factory=dict)
    rule_fired: bool = False

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class ValidationReport:
    """Complete validation report for a rule."""
    rule_name: str = ""
    config_version: str = "0.1.0-exp"

    # Precision/Recall from labeled windows
    precision: float = 0.0
    recall: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    # Seed consistency
    seed_consistency: float = 0.0
    seeds_same_direction: int = 0
    total_seeds: int = 0

    # Intervention effectiveness
    target_metric_improvement: float = 0.0  # relative improvement
    main_metric_improvement: float = 0.0
    non_target_degraded: bool = False
    cost_increase: float = 0.0

    # Acceptance
    meets_precision: bool = False
    meets_recall: bool = False
    meets_seed_consistency: bool = False
    meets_intervention: bool = False
    meets_cost: bool = False
    overall_pass: bool = False

    # Detailed data
    labeled_windows: list[LabeledWindow] = field(default_factory=list)
    seed_results: list[SeedResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# Validation Report: {self.rule_name}",
            f"",
            f"## Precision / Recall",
            f"- Precision: {self.precision:.3f} (threshold: 0.80) {'✓' if self.meets_precision else '✗'}",
            f"- Recall: {self.recall:.3f} (threshold: 0.80) {'✓' if self.meets_recall else '✗'}",
            f"- TP={self.tp}, FP={self.fp}, FN={self.fn}, TN={self.tn}",
            f"",
            f"## Seed Consistency",
            f"- Consistency: {self.seed_consistency:.3f} ({self.seeds_same_direction}/{self.total_seeds} seeds) (threshold: 2/3) {'✓' if self.meets_seed_consistency else '✗'}",
            f"",
            f"## Intervention Effectiveness",
            f"- Target metric improvement: {self.target_metric_improvement:.1f}% (threshold: 20%) {'✓' if self.meets_intervention else '✗'}",
            f"- Main metric improvement: {self.main_metric_improvement:.1f}% (threshold: 3%)",
            f"- Non-target degraded: {self.non_target_degraded}",
            f"- Cost increase: {self.cost_increase:.1f}% (threshold: 10%) {'✓' if self.meets_cost else '✗'}",
            f"",
            f"## Overall: {'PASS ✓' if self.overall_pass else 'FAIL ✗'}",
        ]
        return "\n".join(lines)


# ── Validation functions ─────────────────────────────────────────

def compute_precision_recall(windows: list[LabeledWindow]) -> tuple[float, float, int, int, int, int]:
    """Compute precision and recall from labeled windows.

    Returns (precision, recall, tp, fp, fn, tn).
    """
    tp = sum(1 for w in windows if w.rule_fired and w.anomaly_present)
    fp = sum(1 for w in windows if w.rule_fired and not w.anomaly_present)
    fn = sum(1 for w in windows if not w.rule_fired and w.anomaly_present)
    tn = sum(1 for w in windows if not w.rule_fired and not w.anomaly_present)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall, tp, fp, fn, tn


def compute_seed_consistency(seed_results: list[SeedResult]) -> tuple[float, int, int]:
    """Compute seed consistency: fraction of seeds showing same direction.

    Direction = whether the rule fired (for discovery) or whether intervention
    improved the target metric (for intervention phase).

    Returns (consistency_ratio, same_direction_count, total_seeds).
    """
    if not seed_results:
        return 0.0, 0, 0

    # For intervention: count seeds where target metric improved
    improved = sum(1 for s in seed_results if any(v > 0 for v in s.target_metrics.values()))
    total = len(seed_results)
    # Consistency = max(improved, not_improved) / total
    same_direction = max(improved, total - improved)
    return same_direction / total if total else 0.0, same_direction, total


def compute_intervention_effectiveness(
    baseline_seeds: list[SeedResult],
    intervention_seeds: list[SeedResult],
    target_metric_name: str,
    main_metric_name: Optional[str] = None,
    cost_metric_name: Optional[str] = None,
) -> dict:
    """Compute relative improvement from intervention vs baseline.

    Returns dict with:
    - target_metric_improvement: relative improvement in target metric
    - main_metric_improvement: relative improvement in main metric
    - non_target_degraded: whether any non-target metric degraded
    - cost_increase: relative increase in cost metric
    """
    def mean_metric(seeds: list[SeedResult], name: str) -> float:
        vals = [s.target_metrics.get(name, 0.0) for s in seeds if name in s.target_metrics]
        if not vals:
            vals = [s.non_target_metrics.get(name, 0.0) for s in seeds if name in s.non_target_metrics]
        return sum(vals) / len(vals) if vals else 0.0

    result = {
        "target_metric_improvement": 0.0,
        "main_metric_improvement": 0.0,
        "non_target_degraded": False,
        "cost_increase": 0.0,
    }

    # Target metric
    baseline_target = mean_metric(baseline_seeds, target_metric_name)
    interv_target = mean_metric(intervention_seeds, target_metric_name)
    if baseline_target != 0:
        result["target_metric_improvement"] = (interv_target - baseline_target) / abs(baseline_target) * 100

    # Main metric
    if main_metric_name:
        baseline_main = mean_metric(baseline_seeds, main_metric_name)
        interv_main = mean_metric(intervention_seeds, main_metric_name)
        if baseline_main != 0:
            result["main_metric_improvement"] = (interv_main - baseline_main) / abs(baseline_main) * 100

    # Non-target degradation check
    if baseline_seeds and intervention_seeds:
        for name in baseline_seeds[0].non_target_metrics:
            if name == cost_metric_name or name == target_metric_name or name == main_metric_name:
                continue
            b_val = mean_metric(baseline_seeds, name)
            i_val = mean_metric(intervention_seeds, name)
            if b_val > 0 and i_val < b_val * 0.95:  # > 5% degradation
                result["non_target_degraded"] = True
                break

    # Cost
    if cost_metric_name:
        baseline_cost = mean_metric(baseline_seeds, cost_metric_name)
        interv_cost = mean_metric(intervention_seeds, cost_metric_name)
        if baseline_cost > 0:
            result["cost_increase"] = (interv_cost - baseline_cost) / baseline_cost * 100

    return result


def validate_rule(
    rule_name: str,
    labeled_windows: list[LabeledWindow],
    seed_results: list[SeedResult],
    intervention_data: Optional[dict] = None,
    config_version: str = "0.1.0-exp",
) -> ValidationReport:
    """Run full validation for a diagnostic rule.

    Args:
        rule_name: Name of the rule being validated.
        labeled_windows: Labeled windows for precision/recall computation.
        seed_results: Per-seed results for consistency check.
        intervention_data: Dict with baseline_seeds, intervention_seeds,
                          target_metric_name, etc. for intervention effectiveness.
        config_version: Config version string.

    Returns:
        ValidationReport with all computed metrics and pass/fail status.
    """
    report = ValidationReport(rule_name=rule_name, config_version=config_version)
    report.labeled_windows = labeled_windows
    report.seed_results = seed_results

    # Precision / Recall
    if labeled_windows:
        prec, rec, tp, fp, fn, tn = compute_precision_recall(labeled_windows)
        report.precision = prec
        report.recall = rec
        report.tp, report.fp, report.fn, report.tn = tp, fp, fn, tn
        report.meets_precision = prec >= 0.80
        report.meets_recall = rec >= 0.80

    # Seed consistency
    if seed_results:
        consistency, same_dir, total = compute_seed_consistency(seed_results)
        report.seed_consistency = consistency
        report.seeds_same_direction = same_dir
        report.total_seeds = total
        report.meets_seed_consistency = same_dir >= 2 and total >= 3 and same_dir / total >= 2 / 3

    # Intervention effectiveness
    if intervention_data:
        eff = compute_intervention_effectiveness(
            intervention_data.get("baseline_seeds", []),
            intervention_data.get("intervention_seeds", []),
            intervention_data.get("target_metric_name", ""),
            intervention_data.get("main_metric_name"),
            intervention_data.get("cost_metric_name"),
        )
        report.target_metric_improvement = eff["target_metric_improvement"]
        report.main_metric_improvement = eff["main_metric_improvement"]
        report.non_target_degraded = eff["non_target_degraded"]
        report.cost_increase = eff["cost_increase"]
        report.meets_intervention = (
            eff["target_metric_improvement"] >= 20.0 or
            eff["main_metric_improvement"] >= 3.0
        ) and not eff["non_target_degraded"]
        report.meets_cost = eff["cost_increase"] <= 10.0

    # Overall pass
    report.overall_pass = (
        report.meets_precision and
        report.meets_recall and
        report.meets_seed_consistency and
        report.meets_intervention and
        report.meets_cost
    )

    return report


def update_rule_card(card: RuleCard, report: ValidationReport) -> RuleCard:
    """Update a rule card with validation results."""
    card.validation_precision = report.precision
    card.validation_recall = report.recall
    card.seed_consistency = report.seed_consistency
    if report.overall_pass:
        card.status = "validated"
    return card
