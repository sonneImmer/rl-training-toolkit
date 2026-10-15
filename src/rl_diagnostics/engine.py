"""Unified diagnostics engine that orchestrates all analysis steps."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from typing import Optional

from .schemas.data_models import GroupRecord, RolloutRecord
from .parsers.log_parser import parse_log
from .parsers.rollout_parser import parse_rollout_dir
from .analyzers.rollout_quality import assess_rollout_quality, RolloutQualityResult
from .analyzers.group_signal import (
    diagnose_group_signal, aggregate_step, GroupSignalConfig,
    GroupSignalResult, StepAggregation,
)
from .analyzers.difficulty_analyzer import (
    assign_difficulty, stratify_by_difficulty, load_difficulty_map,
    DifficultyReport,
)
from .exporters.prompt_exporter import export_prompts, ExportFilter
from .analyzers.stability_analyzer import run_stability_diagnosis, StabilityReport
from .config.stability_config import StabilityConfig
from .reporters.html_reporter import generate_html_report


def run_diagnostics(
    log_path: Optional[str] = None,
    script_path: Optional[str] = None,
    rollout_dir: Optional[str] = None,
    difficulty_map_path: Optional[str] = None,
    output_dir: str = "./diagnostics_output",
    experiment_id_override: Optional[str] = None,
    signal_config: Optional[GroupSignalConfig] = None,
    export_filter: Optional[ExportFilter] = None,
    stability_config: Optional[StabilityConfig] = None,
    generate_html: bool = True,
    export_svgs: bool = False,
) -> dict:
    """Run full offline diagnostics pipeline.
    
    Returns a summary dict with counts and file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary = {"steps": [], "output_files": []}

    # ── REQ-01: Parse training log ────────────────────────────────
    exp = None
    metrics = []
    report = None
    epoch_map: dict[int, int] = {}

    if log_path and os.path.isfile(log_path):
        exp, metrics, report = parse_log(log_path, script_path=script_path)
        if experiment_id_override:
            exp.experiment_id = experiment_id_override

        with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
            json.dump(exp.to_dict(), f, indent=2, ensure_ascii=False)
        summary["output_files"].append("experiment_config.json")

        if metrics:
            with open(os.path.join(output_dir, "step_metrics.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "experiment_id", "step", "epoch", "phase",
                    "metric_name", "metric_value", "source_file", "source_line"
                ])
                writer.writeheader()
                for m in metrics:
                    writer.writerow(m.to_dict())
            summary["output_files"].append("step_metrics.csv")

            for m in metrics:
                if m.metric_name == "training/epoch" and m.phase == "train":
                    epoch_map[m.step] = int(m.metric_value)

        if report:
            with open(os.path.join(output_dir, "parse_report.json"), "w") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            summary["output_files"].append("parse_report.json")

    # ── REQ-02: Parse rollout JSONL ───────────────────────────────
    rollouts: list[RolloutRecord] = []
    groups: list[GroupRecord] = []

    if rollout_dir and os.path.isdir(rollout_dir):
        exp_id = exp.experiment_id if exp else (experiment_id_override or "unknown")
        rollouts, groups, warnings = parse_rollout_dir(rollout_dir, exp_id, epoch_map)

        with open(os.path.join(output_dir, "rollouts_normalized.jsonl"), "w") as f:
            for r in rollouts:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        summary["output_files"].append("rollouts_normalized.jsonl")

        with open(os.path.join(output_dir, "groups_normalized.jsonl"), "w") as f:
            for g in groups:
                f.write(json.dumps(g.to_dict(), ensure_ascii=False) + "\n")
        summary["output_files"].append("groups_normalized.jsonl")

        if warnings:
            with open(os.path.join(output_dir, "parse_warnings.json"), "w") as f:
                json.dump(warnings, f, indent=2, ensure_ascii=False)
            summary["output_files"].append("parse_warnings.json")

    # ── Initialize variables for optional REQ-03+ blocks ─────────
    quality_results: list[RolloutQualityResult] = []
    group_signals: list[GroupSignalResult] = []
    step_aggs: list[StepAggregation] = []
    diff_report = DifficultyReport()

    # ── REQ-03: Rollout quality + Group signal ────────────────────
    if groups:
        exp_id = groups[0].experiment_id if groups else "unknown"

        # Layer 1: Assess each rollout quality
        for r in rollouts:
            q = assess_rollout_quality(r)
            quality_results.append(q)

        with open(os.path.join(output_dir, "rollout_quality.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "experiment_id", "step", "group_id", "rollout_id", "status", "reason", "details"
            ])
            writer.writeheader()
            for q in quality_results:
                row = q.to_dict()
                row["details"] = json.dumps(row.get("details", {}))
                writer.writerow(row)
        summary["output_files"].append("rollout_quality.csv")

        # REQ-06: Assign difficulty
        external_map = None
        if difficulty_map_path and os.path.isfile(difficulty_map_path):
            external_map, _ = load_difficulty_map(difficulty_map_path)

        groups, diff_report = assign_difficulty(groups, external_map)

        # Layer 2: Diagnose group learning signal
        if signal_config is None:
            signal_config = GroupSignalConfig()

        for group in groups:
            group_qs = [q for q in quality_results
                        if q.group_id == group.group_id and q.step == group.step]
            gs = diagnose_group_signal(group, group_qs, signal_config)
            group_signals.append(gs)

        with open(os.path.join(output_dir, "group_diagnostics.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "experiment_id", "step", "group_id", "signal_type", "reason",
                "rollout_count", "usable_count", "unique_reward_count",
                "reward_range", "reward_std", "informative_pair_ratio",
                "low_information", "correctness_type", "diagnostic_reason"
            ])
            writer.writeheader()
            for gs in group_signals:
                row = gs.to_dict()
                # Exclude complex fields not suitable for CSV
                row.pop("rewards", None)
                row.pop("thresholds", None)
                writer.writerow(row)
        summary["output_files"].append("group_diagnostics.csv")

        # Aggregate by step
        all_steps = sorted(set(g.step for g in groups))
        for step in all_steps:
            epoch = epoch_map.get(step)
            agg = aggregate_step(step, epoch, group_signals, quality_results)
            step_aggs.append(agg)

        with open(os.path.join(output_dir, "step_aggregations.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[k for k in step_aggs[0].to_dict()] if step_aggs else [])
            if step_aggs:
                writer.writeheader()
                for sa in step_aggs:
                    writer.writerow(sa.to_dict())
        summary["output_files"].append("step_aggregations.csv")

        # REQ-06: Difficulty stratification
        if diff_report.has_difficulty:
            diff_buckets = stratify_by_difficulty(groups, group_signals)
            diff_report.buckets = diff_buckets

        with open(os.path.join(output_dir, "difficulty_report.json"), "w") as f:
            json.dump(diff_report.to_dict(), f, indent=2, ensure_ascii=False)
        summary["output_files"].append("difficulty_report.json")

        # ── REQ-05: Export abnormal prompts ─────────────────────────
        exported_count, blocked_count, blocked_records = export_prompts(
            groups=groups,
            rollouts=rollouts,
            quality_results=quality_results,
            group_signals=group_signals,
            output_path=os.path.join(output_dir, "invalid_prompts.jsonl"),
            export_filter=export_filter,
        )
        summary["output_files"].append("invalid_prompts.jsonl")
        summary["exported_prompts"] = exported_count
        summary["blocked_prompts"] = blocked_count

        if blocked_records:
            with open(os.path.join(output_dir, "export_blocked.json"), "w") as f:
                json.dump(blocked_records, f, indent=2, ensure_ascii=False)
            summary["output_files"].append("export_blocked.json")

        # Summary stats
        summary["total_rollouts"] = len(rollouts)
        summary["total_groups"] = len(groups)
        summary["usable_rollouts"] = sum(1 for q in quality_results if q.status == "usable")
        summary["effective_groups"] = sum(1 for gs in group_signals if gs.signal_type == "effective_group")
        summary["low_info_groups"] = sum(1 for gs in group_signals if gs.low_information)
        summary["steps"] = [{"step": sa.step, "effective_ratio": sa.effective_group_ratio,
                             "low_info_ratio": sa.low_information_group_ratio}
                            for sa in step_aggs]

    # ── REQ-07: Stability diagnosis ────────────────────────────────
    stability_report: Optional[StabilityReport] = None
    if metrics:
        stability_report = run_stability_diagnosis(
            metrics=metrics,
            step_aggs=step_aggs if groups else None,
            config=stability_config,
        )
        with open(os.path.join(output_dir, "stability_report.json"), "w") as f:
            json.dump(stability_report.to_dict(), f, indent=2, ensure_ascii=False)
        summary["output_files"].append("stability_report.json")
        summary["stability_alerts"] = stability_report.total_alerts
        summary["stability_critical"] = stability_report.critical_count

    # ── REQ-11: HTML report ────────────────────────────────────────
    if generate_html:
        html_path = os.path.join(output_dir, "diagnostics_report.html")
        generate_html_report(
            experiment=exp,
            parse_report=report,
            step_aggs=step_aggs if groups else [],
            quality_results=quality_results if groups else [],
            group_signals=group_signals if groups else [],
            stability_report=stability_report,
            diff_report=diff_report if groups else None,
            summary=summary,
            output_path=html_path,
            invalid_prompts_path=os.path.join(output_dir, "invalid_prompts.jsonl"),
        )
        summary["output_files"].append("diagnostics_report.html")

    # ── Export standalone SVG charts ───────────────────────────────
    if export_svgs:
        from .reporters.visualizer import export_all_svgs
        # Load invalid prompts for card chart
        invalid_prompts_data: list[dict] = []
        ip_path = os.path.join(output_dir, "invalid_prompts.jsonl")
        if os.path.isfile(ip_path):
            with open(ip_path, "r") as f:
                for line in f:
                    try:
                        invalid_prompts_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        svg_files = export_all_svgs(
            output_dir=output_dir,
            step_aggs=step_aggs if groups else [],
            quality_results=quality_results if groups else [],
            group_signals=group_signals if groups else [],
            diff_report=diff_report if groups else None,
            invalid_prompts=invalid_prompts_data,
        )
        for sf in svg_files:
            fname = os.path.basename(sf)
            if fname not in summary["output_files"]:
                summary["output_files"].append(fname)

    # ── Write metadata ─────────────────────────────────────────────
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump({
            "experiment_id": exp.experiment_id if exp else "unknown",
            "schema_version": "1.0.0",
            "log_path": log_path,
            "rollout_dir": rollout_dir,
            "output_dir": output_dir,
        }, f, indent=2, ensure_ascii=False)
    summary["output_files"].append("metadata.json")

    return summary
