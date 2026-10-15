"""REQ-12: Experiment matrix configuration.

Defines the experiment matrix for rule validation across discovery and
migration phases. This is the configuration framework — actual experiment
execution requires real training environments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    experiment_id: str = ""
    phase: str = "discovery"  # discovery | migration
    rule_name: str = ""
    model: str = ""
    dataset: str = ""
    seed: int = 0
    intervention: str = "baseline"  # baseline | intervention_a | intervention_b | ...
    intervention_params: dict = field(default_factory=dict)
    training_steps: int = 0
    status: str = "planned"  # planned | running | completed | failed
    metrics_path: str = ""
    rollout_dir: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class ExperimentMatrix:
    """Complete experiment matrix for rule validation.

    Defines all experiment runs needed for phases B (discovery) and D (migration).
    """
    # Discovery phase
    discovery_model: str = ""
    discovery_datasets: list[str] = field(default_factory=list)
    discovery_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    discovery_interventions: list[str] = field(default_factory=lambda: ["baseline"])

    # Migration phase
    migration_model: str = ""
    migration_dataset: str = ""
    migration_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    migration_interventions: list[str] = field(default_factory=lambda: ["baseline"])

    # Rules to validate
    rules: list[str] = field(default_factory=lambda: [
        "low_information_group",
        "output_truncation",
        "kl_drift",
    ])

    # Config
    config_version: str = "0.1.0-exp"
    experiments: list[ExperimentConfig] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    def generate_runs(self) -> list[ExperimentConfig]:
        """Generate all experiment configurations from the matrix.

        Returns a list of ExperimentConfig objects for all planned runs.
        """
        runs: list[ExperimentConfig] = []
        run_id = 0

        # Discovery phase
        for rule in self.rules:
            for dataset in self.discovery_datasets:
                for intervention in self.discovery_interventions:
                    for seed in self.discovery_seeds:
                        run_id += 1
                        runs.append(ExperimentConfig(
                            experiment_id=f"disc_{run_id:03d}",
                            phase="discovery",
                            rule_name=rule,
                            model=self.discovery_model,
                            dataset=dataset,
                            seed=seed,
                            intervention=intervention,
                            status="planned",
                        ))

        # Migration phase
        for rule in self.rules:
            for intervention in self.migration_interventions:
                for seed in self.migration_seeds:
                    run_id += 1
                    runs.append(ExperimentConfig(
                        experiment_id=f"migr_{run_id:03d}",
                        phase="migration",
                        rule_name=rule,
                        model=self.migration_model,
                        dataset=self.migration_dataset,
                        seed=seed,
                        intervention=intervention,
                        status="planned",
                    ))

        self.experiments = runs
        return runs

    def estimate_total_runs(self) -> int:
        """Estimate total number of experiment runs."""
        discovery = len(self.rules) * len(self.discovery_datasets) * len(self.discovery_interventions) * len(self.discovery_seeds)
        migration = len(self.rules) * len(self.migration_interventions) * len(self.migration_seeds)
        return discovery + migration

    def to_markdown(self) -> str:
        """Render the experiment matrix as markdown."""
        total = self.estimate_total_runs()
        lines = [
            "# Experiment Matrix",
            f"",
            f"**Config Version:** {self.config_version}",
            f"**Estimated Total Runs:** {total}",
            f"",
            f"## Discovery Phase (B)",
            f"- **Model:** {self.discovery_model or 'TBD'}",
            f"- **Datasets:** {', '.join(self.discovery_datasets) or 'TBD'}",
            f"- **Seeds:** {', '.join(str(s) for s in self.discovery_seeds)}",
            f"- **Interventions:** {', '.join(self.discovery_interventions)}",
            f"- **Rules:** {', '.join(self.rules)}",
            f"- **Runs:** {len(self.rules) * len(self.discovery_datasets) * len(self.discovery_interventions) * len(self.discovery_seeds)}",
            f"",
            f"## Migration Phase (D)",
            f"- **Model:** {self.migration_model or 'TBD'}",
            f"- **Dataset:** {self.migration_dataset or 'TBD'}",
            f"- **Seeds:** {', '.join(str(s) for s in self.migration_seeds)}",
            f"- **Interventions:** {', '.join(self.migration_interventions)}",
            f"- **Runs:** {len(self.rules) * len(self.migration_interventions) * len(self.migration_seeds)}",
            f"",
            f"## Acceptance Criteria",
            f"- Precision >= 0.80 on labeled windows",
            f"- Recall >= 0.80 on labeled windows",
            f"- >= 2/3 seeds show same direction",
            f"- Target metric relative improvement >= 20%, or main metric >= 3%",
            f"- Core non-target metrics show no significant degradation",
            f"- Training cost increase <= 10%",
            f"- Migration: >= 2/3 seeds maintain direction",
            f"",
            f"## Generated Runs",
        ]
        for exp in self.experiments:
            lines.append(
                f"- `{exp.experiment_id}` [{exp.phase}] {exp.rule_name} | "
                f"{exp.model} | {exp.dataset} | seed={exp.seed} | {exp.intervention} | {exp.status}"
            )
        return "\n".join(lines)


def create_default_matrix() -> ExperimentMatrix:
    """Create the default experiment matrix from REQ-12 spec."""
    return ExperimentMatrix(
        discovery_model="qwen2.5-7b",
        discovery_datasets=["gsm8k", "math"],
        discovery_seeds=[0, 1, 2],
        discovery_interventions=["baseline", "intervention_a", "intervention_b", "intervention_c"],
        migration_model="qwen2.5-14b",
        migration_dataset="unseen_dataset",
        migration_seeds=[0, 1, 2],
        migration_interventions=["baseline", "intervention_a", "intervention_b", "intervention_c"],
        rules=["low_information_group", "output_truncation", "kl_drift"],
    )
