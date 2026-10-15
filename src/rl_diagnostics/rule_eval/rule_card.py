"""REQ-12: Rule card data model and registry.

Each diagnostic rule forms a rule card containing its definition,
validation status, evidence, and migration results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InterventionRecord:
    """Record of an intervention applied to test a rule."""
    name: str = ""
    description: str = ""
    before_metrics: dict[str, float] = field(default_factory=dict)
    after_metrics: dict[str, float] = field(default_factory=dict)
    relative_change: dict[str, float] = field(default_factory=dict)
    seeds: list[int] = field(default_factory=list)
    seed_consistent: bool = False
    significant: bool = False

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class MigrationResult:
    """Migration validation result for a rule."""
    dataset: str = ""
    model: str = ""
    seeds: list[int] = field(default_factory=list)
    direction_consistent: bool = False
    effect_size: float = 0.0
    effect_retention: float = 0.0  # retained effect relative to discovery phase
    notes: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class RuleCard:
    """Complete rule card for a diagnostic rule.

    Contains the rule definition, evidence, interventions, and migration
    validation results. Rules without completed experiments are marked
    'experimental'.
    """
    # Identity
    rule_name: str = ""
    rule_version: str = "0.1.0"
    config_version: str = "0.1.0-exp"

    # Applicability
    applicable_algorithms: list[str] = field(default_factory=list)
    trigger_condition: str = ""
    evidence_window: str = ""

    # Evidence
    typical_badcases: list[str] = field(default_factory=list)
    candidate_causes: list[str] = field(default_factory=list)
    suggested_interventions: list[str] = field(default_factory=list)

    # Validation
    status: str = "experimental"  # experimental | validated | deprecated
    validation_precision: Optional[float] = None
    validation_recall: Optional[float] = None
    seed_consistency: Optional[float] = None  # fraction of seeds with same direction
    intervention_records: list[InterventionRecord] = field(default_factory=list)

    # Migration
    migration_results: list[MigrationResult] = field(default_factory=list)
    validated_datasets: list[str] = field(default_factory=list)
    validated_models: list[str] = field(default_factory=list)

    # Limitations
    not_applicable_conditions: list[str] = field(default_factory=list)
    known_counterexamples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    def to_markdown(self) -> str:
        """Render rule card as markdown for documentation."""
        lines = [
            f"# Rule Card: {self.rule_name}",
            f"",
            f"**Version:** {self.rule_version}  ",
            f"**Config Version:** {self.config_version}  ",
            f"**Status:** {self.status}",
            f"",
            f"## Applicability",
            f"- **Algorithms:** {', '.join(self.applicable_algorithms) or 'N/A'}",
            f"- **Trigger Condition:** {self.trigger_condition or 'N/A'}",
            f"- **Evidence Window:** {self.evidence_window or 'N/A'}",
            f"",
            f"## Evidence",
            f"- **Typical Badcases:**",
        ]
        for bc in self.typical_badcases:
            lines.append(f"  - {bc}")
        lines.append(f"- **Candidate Causes:**")
        for c in self.candidate_causes:
            lines.append(f"  - {c}")
        lines.append(f"- **Suggested Interventions:**")
        for iv in self.suggested_interventions:
            lines.append(f"  - {iv}")

        lines.append(f"")
        lines.append(f"## Validation")
        if self.validation_precision is not None:
            lines.append(f"- **Precision:** {self.validation_precision:.3f}")
        if self.validation_recall is not None:
            lines.append(f"- **Recall:** {self.validation_recall:.3f}")
        if self.seed_consistency is not None:
            lines.append(f"- **Seed Consistency:** {self.seed_consistency:.3f}")

        if self.intervention_records:
            lines.append(f"- **Intervention Records:**")
            for ir in self.intervention_records:
                lines.append(f"  - {ir.name}: consistent={ir.seed_consistent}, significant={ir.significant}")

        lines.append(f"")
        lines.append(f"## Migration")
        if self.migration_results:
            for mr in self.migration_results:
                lines.append(
                    f"- {mr.dataset}/{mr.model}: direction_consistent={mr.direction_consistent}, "
                    f"effect_retention={mr.effect_retention:.3f}"
                )
        else:
            lines.append(f"- No migration results yet")

        lines.append(f"")
        lines.append(f"## Limitations")
        for na in self.not_applicable_conditions:
            lines.append(f"- **Not applicable when:** {na}")
        for ce in self.known_counterexamples:
            lines.append(f"- **Known counterexample:** {ce}")

        lines.append(f"")
        lines.append(f"## Validated Datasets & Models")
        lines.append(f"- Datasets: {', '.join(self.validated_datasets) or 'None'}")
        lines.append(f"- Models: {', '.join(self.validated_models) or 'None'}")

        return "\n".join(lines)


# ── Rule Registry ────────────────────────────────────────────────

class RuleRegistry:
    """Registry for diagnostic rule cards."""

    def __init__(self):
        self._rules: dict[str, RuleCard] = {}

    def register(self, card: RuleCard) -> None:
        self._rules[card.rule_name] = card

    def get(self, name: str) -> Optional[RuleCard]:
        return self._rules.get(name)

    def all_rules(self) -> list[RuleCard]:
        return list(self._rules.values())

    def validated_rules(self) -> list[RuleCard]:
        return [r for r in self._rules.values() if r.status == "validated"]

    def experimental_rules(self) -> list[RuleCard]:
        return [r for r in self._rules.values() if r.status == "experimental"]

    def to_dict(self) -> dict:
        return {name: card.to_dict() for name, card in self._rules.items()}

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def create_default_rules() -> RuleRegistry:
    """Create a registry with the three default rules from REQ-12.

    These are the three rules suggested for strict validation:
    1. Low information group
    2. Output truncation
    3. KL drift / update too strong
    """
    registry = RuleRegistry()

    registry.register(RuleCard(
        rule_name="low_information_group",
        rule_version="0.1.0",
        applicable_algorithms=["GRPO", "PPO", "REINFORCE"],
        trigger_condition="All rollouts in a group have equal or near-equal reward (within epsilon)",
        evidence_window="single_step",
        typical_badcases=[
            "All responses get reward 0 (model cannot solve the problem)",
            "All responses get reward 1 (problem is too easy, no learning signal)",
            "Reward variance within group below threshold for consecutive steps",
        ],
        candidate_causes=[
            "Problem difficulty mismatch with model capability",
            "Reward function not discriminative enough",
            "Data quality issues in the prompt/ground truth",
        ],
        suggested_interventions=[
            "Filter or re-weight low-information prompts in training batch",
            "Adjust reward function granularity",
            "Curriculum learning: sort by difficulty",
        ],
        not_applicable_conditions=[
            "rollout.n = 1 (single rollout, no group signal)",
            "Deterministic decoding (no variance expected)",
        ],
    ))

    registry.register(RuleCard(
        rule_name="output_truncation",
        rule_version="0.1.0",
        applicable_algorithms=["GRPO", "PPO", "REINFORCE"],
        trigger_condition="finish_reason = 'length' or response_token_length >= max_response_length for > threshold% of rollouts",
        evidence_window="sliding_window_3_steps",
        typical_badcases=[
            "Model generates repetitive loops until max length",
            "Model output cut off mid-reasoning",
            "Response length consistently at ceiling",
        ],
        candidate_causes=[
            "Max response length too short for task complexity",
            "Model has not learned to produce concise answers",
            "Decoding temperature too high causing repetitive generation",
        ],
        suggested_interventions=[
            "Increase max_response_length",
            "Add length penalty to reward",
            "Lower decoding temperature",
        ],
        not_applicable_conditions=[
            "Short-answer tasks where max length is rarely reached",
        ],
    ))

    registry.register(RuleCard(
        rule_name="kl_drift",
        rule_version="0.1.0",
        applicable_algorithms=["GRPO", "PPO"],
        trigger_condition="actor/kl_loss exceeds threshold or kl_divergence shows sustained upward trend",
        evidence_window="sliding_window_5_steps",
        typical_badcases=[
            "KL loss spikes after learning rate warmup",
            "Policy diverges from reference, causing reward collapse",
            "Sustained KL increase with decreasing reward",
        ],
        candidate_causes=[
            "Learning rate too high",
            "KL coefficient too low",
            "Insufficient warmup steps",
        ],
        suggested_interventions=[
            "Reduce learning rate",
            "Increase KL coefficient",
            "Extend warmup schedule",
        ],
        not_applicable_conditions=[
            "REINFORCE (no KL term by design)",
            "First few steps of warmup (KL increase is expected)",
        ],
    ))

    return registry
