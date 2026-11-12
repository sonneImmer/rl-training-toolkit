#!/usr/bin/env python3
"""Build an evidence-bound GRPO/NGRPO/PPO comparison report."""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path

from rl_diagnostics.experiment_store import ExperimentStore
from rl_diagnostics.parsers.log_parser import parse_log


METRICS = (
    "actor/entropy",
    "actor/kl_loss",
    "actor/ppo_kl",
    "actor/pg_clipfrac",
    "actor/grad_norm",
    "critic/vf_loss",
    "critic/vf_clipfrac",
    "critic/grad_norm",
    "timing_s/step",
    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/update_critic",
    "perf/throughput",
)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def metric_means(log_path: Path) -> dict[str, float | None]:
    _, metrics, _ = parse_log(str(log_path))
    by_name = {name: [] for name in METRICS}
    for metric in metrics:
        if metric.metric_name in by_name and metric.phase == "train":
            by_name[metric.metric_name].append(metric.metric_value)
    return {name: mean(values) for name, values in by_name.items()}


def summarize(run_dir: Path, state_dir: Path) -> dict:
    store = ExperimentStore(run_dir.parent, state_dir, stale_seconds=999999)
    snapshot = store.refresh_run(run_dir)
    steps = snapshot.get("steps", [])
    mechanism = snapshot.get("diagnostics", {}).get("mechanism", {})
    stability = snapshot.get("diagnostics", {}).get("stability", {})
    evaluations = snapshot.get("evaluations", [])
    evaluation = evaluations[-1] if evaluations else {}
    return {
        "run_dir": str(run_dir.resolve()),
        "experiment_id": snapshot.get("experiment_id"),
        "algorithm": snapshot.get("config", {}).get("algorithm"),
        "advantage_estimator": mechanism.get("algorithm_profile", {}).get("advantage_estimator"),
        "status": snapshot.get("status"),
        "steps": len(steps),
        "latest_step": snapshot.get("progress", {}).get("current_step"),
        "reward_mean": mean([row["reward_mean"] for row in steps if row.get("reward_mean") is not None]),
        "effective_group_ratio": mean([row["effective_group_ratio"] for row in steps if row.get("effective_group_ratio") is not None]),
        "low_information_group_ratio": mean([row["low_information_group_ratio"] for row in steps if row.get("low_information_group_ratio") is not None]),
        "advantage_coverage_ratio": mean([row["advantage_coverage_ratio"] for row in steps if row.get("advantage_coverage_ratio") is not None]),
        "homogeneous_zero_advantage_groups": sum(row.get("homogeneous_reward_zero_advantage_groups", 0) for row in steps),
        "homogeneous_nonzero_advantage_groups": sum(row.get("homogeneous_reward_nonzero_advantage_groups", 0) for row in steps),
        "all_wrong_negative_advantage_groups": sum(row.get("all_wrong_negative_advantage_groups", 0) for row in steps),
        "metric_means": metric_means(run_dir / "train.log"),
        "stability_triggered": [
            item.get("rule_name") for item in stability.get("evaluations", []) if item.get("status") == "triggered"
        ],
        "mechanism_cards": [
            {"rule_id": item.get("rule_id"), "status": item.get("status"), "observation": item.get("observation")}
            for item in mechanism.get("cards", [])
        ],
        "evaluation": evaluation,
    }


def distinction_checks(rows: list[dict]) -> list[dict]:
    by_algorithm = {row["algorithm"]: row for row in rows}
    checks = []
    grpo = by_algorithm.get("grpo")
    ngrpo = by_algorithm.get("ngrpo")
    ppo = by_algorithm.get("ppo")
    if grpo and ngrpo:
        checks.append({
            "name": "homogeneous_reward_signal_difference",
            "detected": (
                grpo["homogeneous_zero_advantage_groups"] > 0
                and ngrpo["homogeneous_nonzero_advantage_groups"] > 0
            ),
            "evidence": {
                "grpo_homogeneous_zero": grpo["homogeneous_zero_advantage_groups"],
                "ngrpo_homogeneous_nonzero": ngrpo["homogeneous_nonzero_advantage_groups"],
            },
            "boundary": "This detects different recorded advantage behavior, not superior task quality.",
        })
    if ppo:
        checks.append({
            "name": "ppo_critic_path_detected",
            "detected": (
                ppo.get("advantage_estimator") == "gae"
                and ppo["metric_means"].get("critic/vf_loss") is not None
                and ppo["metric_means"].get("timing_s/update_critic") is not None
            ),
            "evidence": {
                "advantage_estimator": ppo.get("advantage_estimator"),
                "critic_vf_loss_mean": ppo["metric_means"].get("critic/vf_loss"),
                "critic_update_seconds_mean": ppo["metric_means"].get("timing_s/update_critic"),
            },
            "boundary": "Critic activity proves a PPO/GAE path was executed, not that PPO improved the model.",
        })
    return checks


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown(report: dict) -> str:
    lines = [
        "# AMLSim GRPO/NGRPO/PPO Controlled Comparison",
        "",
        "All rows should use the same model, dataset, seed, rollout count, and training-step budget. Training evidence and independent evaluation are reported separately.",
        "",
        "| Algorithm | Estimator | Steps | Reward Mean | Effective Group | Low-information Group | Homogeneous Zero Adv | Homogeneous Nonzero Adv | Step Time(s) | Critic Update(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["experiments"]:
        metrics = row["metric_means"]
        lines.append(
            f"| {row['algorithm']} | {row['advantage_estimator']} | {row['latest_step']} | "
            f"{fmt(row['reward_mean'])} | {fmt(row['effective_group_ratio'])} | {fmt(row['low_information_group_ratio'])} | "
            f"{row['homogeneous_zero_advantage_groups']} | {row['homogeneous_nonzero_advantage_groups']} | "
            f"{fmt(metrics.get('timing_s/step'), 2)} | {fmt(metrics.get('timing_s/update_critic'), 2)} |"
        )
    lines.extend(["", "## Tool Distinction Checks", ""])
    for check in report["distinction_checks"]:
        lines.append(f"- **{check['name']}**: {'DETECTED' if check['detected'] else 'NOT DETECTED'}; evidence `{json.dumps(check['evidence'], ensure_ascii=False)}`. {check['boundary']}")
    lines.extend(["", "## Independent Evaluation", ""])
    for row in report["experiments"]:
        evaluation = row.get("evaluation") or {}
        lines.append(f"- **{row['algorithm']}**: `{json.dumps(evaluation, ensure_ascii=False) if evaluation else 'not attached'}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rl-comparison-state-") as state:
        experiments = [summarize(path.resolve(), Path(state)) for path in args.run_dir]
    experiments.sort(key=lambda item: ("grpo", "ngrpo", "ppo").index(item["algorithm"]) if item["algorithm"] in {"grpo", "ngrpo", "ppo"} else 99)
    report = {"schema_version": "1.0.0", "experiments": experiments}
    report["distinction_checks"] = distinction_checks(experiments)
    (args.output_dir / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "comparison.md").write_text(markdown(report), encoding="utf-8")
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
