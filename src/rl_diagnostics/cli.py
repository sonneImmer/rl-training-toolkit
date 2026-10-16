"""CLI entry point for RL diagnostics tool."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from .engine import run_diagnostics
from .parsers.log_parser import parse_log
from .parsers.rollout_parser import parse_rollout_dir
from .analyzers.group_signal import GroupSignalConfig
from .exporters.prompt_exporter import ExportFilter
from .config.stability_config import StabilityConfig
from .reporters.html_reporter import generate_html_report
from .evaluators.math_evaluator import evaluate_math, MathEvalConfig
from .evaluators.code_evaluator import evaluate_code, CodeEvalConfig
from .evaluators.risk_evaluator import evaluate_risk, RiskEvalConfig
from .rule_eval.rule_card import create_default_rules
from .rule_eval.experiment_matrix import create_default_matrix
from .rule_eval.rule_validator import validate_rule, LabeledWindow, SeedResult
from .rule_eval.migration_report import generate_migration_report


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run full offline analysis."""
    signal_config = GroupSignalConfig(
        abs_epsilon=args.abs_epsilon,
        rel_epsilon=args.rel_epsilon,
        min_usable_rollouts=args.min_usable,
        min_effective_ratio=args.min_effective_ratio,
        extreme_ratio_threshold=args.extreme_ratio_threshold,
    )

    export_filter = ExportFilter(
        steps=[int(s) for s in args.filter_steps.split(",")] if args.filter_steps else None,
        diagnosis_types=args.filter_types.split(",") if args.filter_types else None,
        difficulties=args.filter_difficulties.split(",") if args.filter_difficulties else None,
    )

    summary = run_diagnostics(
        log_path=args.log,
        script_path=args.script,
        rollout_dir=args.rollout_dir,
        difficulty_map_path=args.difficulty_map,
        output_dir=args.output_dir,
        experiment_id_override=args.experiment_id,
        signal_config=signal_config,
        export_filter=export_filter,
        generate_html=not args.no_html,
        export_svgs=getattr(args, "export_svgs", False),
    )

    print(f"\n{'='*50}")
    print(f"Diagnostics complete. Output: {args.output_dir}")
    print(f"{'='*50}")
    if "total_rollouts" in summary:
        print(f"  Total rollouts: {summary['total_rollouts']}")
        print(f"  Usable rollouts: {summary['usable_rollouts']}")
        print(f"  Total groups: {summary['total_groups']}")
        print(f"  Effective groups: {summary['effective_groups']}")
        print(f"  Low-info groups: {summary['low_info_groups']}")
        print(f"  Exported prompts: {summary.get('exported_prompts', 0)}")
        print(f"  Blocked prompts: {summary.get('blocked_prompts', 0)}")
    for s in summary.get("steps", []):
        print(f"  Step {s['step']}: effective={s['effective_ratio']:.2f}, low_info={s['low_info_ratio']:.2f}")
    print(f"\n  Output files: {', '.join(summary.get('output_files', []))}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate log and JSONL schema."""
    errors = []

    if args.log and os.path.isfile(args.log):
        exp, metrics, report = parse_log(args.log, script_path=args.script)
        print(f"Log: {args.log}")
        print(f"  Experiment ID: {exp.experiment_id}")
        print(f"  Total metrics: {len(metrics)}")
        print(f"  Parse errors: {report.failed_lines}")
        if report.failed_lines > 0:
            errors.append(f"Log has {report.failed_lines} parse failures")
        if not exp.model:
            errors.append("Model path not found in log")
        if not exp.algorithm:
            errors.append("Algorithm not found in log")

    if args.rollout_dir and os.path.isdir(args.rollout_dir):
        exp_id = exp.experiment_id if exp and metrics else "validate"
        rollouts, groups, warnings = parse_rollout_dir(args.rollout_dir, exp_id)
        print(f"\nRollout dir: {args.rollout_dir}")
        print(f"  Rollouts: {len(rollouts)}")
        print(f"  Groups: {len(groups)}")
        json_errors = sum(1 for r in rollouts if r.parse_status == "json_error")
        schema_errors = sum(1 for r in rollouts if r.parse_status == "schema_error")
        if json_errors:
            errors.append(f"{json_errors} JSON decode errors")
        if schema_errors:
            errors.append(f"{schema_errors} schema errors")

    if errors:
        print("\nValidation FAILED:")
        for e in errors:
            print(f"  ✗ {e}")
        return 1
    else:
        print("\nValidation PASSED ✓")
        return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export abnormal prompts only (from existing diagnostics output)."""
    # Run full analysis first
    summary = run_diagnostics(
        log_path=args.log,
        rollout_dir=args.rollout_dir,
        output_dir=args.output_dir,
    )
    print(f"Exported {summary.get('exported_prompts', 0)} prompts to {args.output_dir}/invalid_prompts.jsonl")
    if summary.get("blocked_prompts", 0) > 0:
        print(f"Blocked {summary['blocked_prompts']} prompts (unrecoverable)")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate HTML report from existing diagnostics output directory."""
    output_dir = args.output_dir
    html_path = args.html_path or os.path.join(output_dir, "diagnostics_report.html")

    if not os.path.isdir(output_dir):
        print(f"Error: output directory not found: {output_dir}")
        return 1

    # Load existing JSON/CSV outputs
    import csv
    from .schemas.data_models import ExperimentRecord, ParseReport
    from .analyzers.group_signal import StepAggregation, GroupSignalResult, GroupSignalConfig
    from .analyzers.rollout_quality import RolloutQualityResult
    from .analyzers.stability_analyzer import StabilityReport, StabilityAlert
    from .analyzers.difficulty_analyzer import DifficultyReport, DifficultyStats
    from .config.stability_config import StabilityConfig

    exp = None
    report = None
    step_aggs = []
    quality_results = []
    group_signals = []
    stability_report = None
    diff_report = None
    summary = {}

    # experiment_config.json
    exp_path = os.path.join(output_dir, "experiment_config.json")
    if os.path.isfile(exp_path):
        with open(exp_path) as f:
            d = json.load(f)
        exp = ExperimentRecord(**{k: v for k, v in d.items() if k in ExperimentRecord.__dataclass_fields__})

    # step_aggregations.csv
    agg_path = os.path.join(output_dir, "step_aggregations.csv")
    if os.path.isfile(agg_path):
        with open(agg_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sa = StepAggregation(
                    step=int(row.get("step", 0)),
                    epoch=int(row["epoch"]) if row.get("epoch") else None,
                    total_groups=int(row.get("total_groups", 0)),
                    analyzable_groups=int(row.get("analyzable_groups", 0)),
                    effective_groups=int(row.get("effective_groups", 0)),
                    constant_score_groups=int(row.get("constant_score_groups", 0)),
                    near_constant_groups=int(row.get("near_constant_groups", 0)),
                    insufficient_groups=int(row.get("insufficient_groups", 0)),
                    unresolved_groups=int(row.get("unresolved_groups", 0)),
                    total_rollouts=int(row.get("total_rollouts", 0)),
                    usable_rollouts=int(row.get("usable_rollouts", 0)),
                    rollout_usable_ratio=float(row.get("rollout_usable_ratio", 0)),
                    effective_group_ratio=float(row.get("effective_group_ratio", 0)),
                    low_information_group_ratio=float(row.get("low_information_group_ratio", 0)),
                    constant_score_group_ratio=float(row.get("constant_score_group_ratio", 0)),
                    near_constant_group_ratio=float(row.get("near_constant_group_ratio", 0)),
                    mean_informative_pair_ratio=float(row.get("mean_informative_pair_ratio", 0)),
                )
                step_aggs.append(sa)

    # rollout_quality.csv
    q_path = os.path.join(output_dir, "rollout_quality.csv")
    if os.path.isfile(q_path):
        with open(q_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                quality_results.append(RolloutQualityResult(
                    experiment_id=row.get("experiment_id", ""),
                    step=int(row.get("step", 0)),
                    group_id=row.get("group_id", ""),
                    rollout_id=int(row.get("rollout_id", 0)),
                    status=row.get("status", "usable"),
                    reason=row.get("reason", ""),
                ))

    # stability_report.json
    stab_path = os.path.join(output_dir, "stability_report.json")
    if os.path.isfile(stab_path):
        with open(stab_path) as f:
            d = json.load(f)
        stability_report = StabilityReport(
            config_version=d.get("config_version", ""),
            total_alerts=d.get("total_alerts", 0),
            critical_count=d.get("critical_count", 0),
            warning_count=d.get("warning_count", 0),
            info_count=d.get("info_count", 0),
            alerts=[StabilityAlert(**a) for a in d.get("alerts", [])],
            rules_checked=d.get("rules_checked", []),
            rules_skipped=d.get("rules_skipped", []),
            metrics_available=d.get("metrics_available", []),
        )

    # difficulty_report.json
    diff_path = os.path.join(output_dir, "difficulty_report.json")
    if os.path.isfile(diff_path):
        with open(diff_path) as f:
            d = json.load(f)
        diff_report = DifficultyReport(
            has_difficulty=d.get("has_difficulty", False),
            missing_rate=d.get("missing_rate", 0),
            total_groups=d.get("total_groups", 0),
            buckets={k: DifficultyStats(**v) for k, v in d.get("buckets", {}).items()},
        )

    # metadata
    meta_path = os.path.join(output_dir, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            summary = json.load(f)

    invalid_prompts_path = os.path.join(output_dir, "invalid_prompts.jsonl")

    # Load domain evaluation results if available
    eval_results = {}
    for domain, fname in [("math", "math_eval_result.json"),
                          ("code", "code_eval_result.json"),
                          ("risk", "risk_eval_result.json")]:
        eval_path = os.path.join(output_dir, fname)
        if os.path.isfile(eval_path):
            with open(eval_path) as f:
                eval_results[domain] = json.load(f)

    generate_html_report(
        experiment=exp,
        parse_report=report,
        step_aggs=step_aggs,
        quality_results=quality_results,
        group_signals=group_signals,
        stability_report=stability_report,
        diff_report=diff_report,
        summary=summary,
        output_path=html_path,
        invalid_prompts_path=invalid_prompts_path,
        eval_results=eval_results if eval_results else None,
    )
    print(f"HTML report generated: {html_path}")
    if getattr(args, "export_svgs", False):
        from .reporters.visualizer import export_all_svgs
        # Load group_signals from group_diagnostics.csv
        group_signals_for_svg: list[GroupSignalResult] = []
        gd_path = os.path.join(output_dir, "group_diagnostics.csv")
        if os.path.isfile(gd_path):
            with open(gd_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gs = GroupSignalResult(
                        experiment_id=row.get("experiment_id", ""),
                        step=int(row.get("step", 0)),
                        group_id=row.get("group_id", ""),
                        signal_type=row.get("signal_type", ""),
                        reason=row.get("reason", ""),
                        rollout_count=int(row.get("rollout_count", 0)),
                        usable_count=int(row.get("usable_count", 0)),
                        unique_reward_count=int(row.get("unique_reward_count", 0)),
                        reward_range=float(row.get("reward_range", 0)),
                        reward_std=float(row.get("reward_std", 0)),
                        informative_pair_ratio=float(row.get("informative_pair_ratio", 0)),
                        low_information=row.get("low_information", "").lower() == "true",
                        diagnostic_reason=row.get("diagnostic_reason", ""),
                    )
                    group_signals_for_svg.append(gs)
        # Load invalid prompts
        invalid_prompts_data: list[dict] = []
        if os.path.isfile(invalid_prompts_path):
            with open(invalid_prompts_path) as f:
                for line in f:
                    try:
                        invalid_prompts_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        svg_files = export_all_svgs(
            output_dir=output_dir,
            step_aggs=step_aggs,
            quality_results=quality_results,
            group_signals=group_signals_for_svg,
            diff_report=diff_report,
            invalid_prompts=invalid_prompts_data,
        )
        for sf in svg_files:
            print(f"  SVG exported: {sf}")
    return 0



def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run domain evaluation (math, code, risk) on rollout data."""
    import csv

    # Load rollouts from normalized JSONL or parse from directory
    responses = []
    ground_truths = []
    rewards = []
    test_cases_data = []

    if args.rollout_dir and os.path.isdir(args.rollout_dir):
        # Parse rollout directory
        exp_id = args.experiment_id or "eval"
        rollouts, groups, warnings = parse_rollout_dir(args.rollout_dir, exp_id)
        for r in rollouts:
            if r.response:
                responses.append(r.response)
            if r.ground_truth:
                ground_truths.append(r.ground_truth)
            if r.training_reward is not None:
                rewards.append(r.training_reward)
    elif args.rollouts_jsonl and os.path.isfile(args.rollouts_jsonl):
        with open(args.rollouts_jsonl) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("response"):
                        responses.append(r["response"])
                    if r.get("ground_truth"):
                        ground_truths.append(r["ground_truth"])
                    if r.get("training_reward") is not None:
                        rewards.append(r["training_reward"])
    else:
        print("Error: need --rollout-dir or --rollouts-jsonl")
        return 1

    if not responses:
        print("Error: no responses found in input data")
        return 1

    # Pad rewards if not enough
    while len(rewards) < len(responses):
        rewards.append(None)
    rewards_list = rewards if any(r is not None for r in rewards) else None

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    domain = args.domain
    print(f"\nRunning {domain} evaluation on {len(responses)} samples...")

    if domain == "math":
        config = MathEvalConfig(
            dataset_type=args.dataset_type,
            tolerance=args.tolerance,
            pass_k=args.pass_k,
        )
        result = evaluate_math(responses, ground_truths, rewards=rewards_list, config=config)
        result_file = os.path.join(output_dir, "math_eval_result.json")
        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"  Parse success rate: {result.parse_success_rate:.3f}")
        print(f"  Exact match: {result.exact_match:.3f}")
        print(f"  Equivalent accuracy: {result.equivalent_accuracy:.3f}")
        print(f"  Pass@1: {result.pass_at_1:.3f}")
        if result.pass_at_k is not None:
            print(f"  Pass@{result.pass_k_value}: {result.pass_at_k:.3f}")
        if result.reward_eval_agreement is not None:
            print(f"  Reward-eval agreement: {result.reward_eval_agreement:.3f}")
        print(f"  Badcases: {result.badcase_counts}")
        print(f"  Result saved to {result_file}")

    elif domain == "code":
        # Load test cases
        test_cases = []
        if args.test_cases and os.path.isfile(args.test_cases):
            with open(args.test_cases) as f:
                test_cases = json.load(f)
        else:
            # Create empty test case list
            test_cases = [[] for _ in responses]

        config = CodeEvalConfig(
            dataset_type=args.dataset_type,
            pass_k=args.pass_k,
            timeout_seconds=args.timeout,
            enable_sandbox=not args.no_sandbox,
        )
        result = evaluate_code(responses, test_cases, rewards=rewards_list, config=config)
        result_file = os.path.join(output_dir, "code_eval_result.json")
        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        if result.blocked:
            print(f"  BLOCKED: {result.blocked_reason}")
        print(f"  Extraction success rate: {result.extraction_success_rate:.3f}")
        print(f"  Compile success rate: {result.compile_success_rate:.3f}")
        print(f"  Runtime success rate: {result.runtime_success_rate:.3f}")
        print(f"  Test pass rate: {result.test_pass_rate:.3f}")
        print(f"  Pass@1: {result.pass_at_1:.3f}")
        print(f"  Timeout rate: {result.timeout_rate:.3f}")
        print(f"  Exception rate: {result.exception_rate:.3f}")
        print(f"  Unsafe API rate: {result.unsafe_api_rate:.3f}")
        if result.reward_eval_agreement is not None:
            print(f"  Reward-eval agreement: {result.reward_eval_agreement:.3f}")
        print(f"  Badcases: {result.badcase_counts}")
        print(f"  Result saved to {result_file}")

    elif domain == "risk":
        # Load labels and predictions
        labels = []
        predictions = []
        scores = None
        if args.eval_data and os.path.isfile(args.eval_data):
            with open(args.eval_data) as f:
                data = json.load(f)
            labels = data.get("labels", [])
            predictions = data.get("predictions", [])
            scores = data.get("scores")
        else:
            # Use ground_truth as labels, rewards as predictions
            labels = [1 if gt and gt != "0" else 0 for gt in ground_truths]
            predictions = [1 if r and r >= 0.5 else 0 for r in rewards if r is not None]

        config = RiskEvalConfig(
            positive_label=args.positive_label,
            has_probability=args.has_probability,
            top_k=args.top_k,
        )
        result = evaluate_risk(labels, predictions, scores=scores, rewards=rewards_list, config=config)
        result_file = os.path.join(output_dir, "risk_eval_result.json")
        with open(result_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        if result.degenerate:
            print(f"  DEGENERATE: {result.degenerate_reason}")
        print(f"  Precision: {result.precision:.3f}")
        print(f"  Recall: {result.recall:.3f}")
        print(f"  F1: {result.f1:.3f}")
        print(f"  F2: {result.f2:.3f}")
        print(f"  PR-AUC: {result.pr_auc:.3f}")
        print(f"  ROC-AUC: {result.roc_auc:.3f}")
        print(f"  KS: {result.ks:.3f}")
        if result.reward_eval_agreement is not None:
            print(f"  Reward-eval agreement: {result.reward_eval_agreement:.3f}")
        print(f"  Result saved to {result_file}")

    else:
        print(f"Error: unknown domain '{domain}'. Use math, code, or risk.")
        return 1

    return 0


def cmd_rule_eval(args: argparse.Namespace) -> int:
    """Run rule validation and generate rule cards / migration reports."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Create default registry and matrix
    registry = create_default_rules()
    matrix = create_default_matrix()

    # Generate experiment matrix
    if args.show_matrix:
        matrix.generate_runs()
        matrix_path = os.path.join(output_dir, "experiment_matrix.md")
        with open(matrix_path, "w") as f:
            f.write(matrix.to_markdown())
        matrix_json_path = os.path.join(output_dir, "experiment_matrix.json")
        with open(matrix_json_path, "w") as f:
            json.dump(matrix.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Experiment matrix ({matrix.estimate_total_runs()} runs) saved to {matrix_path}")
        print(f"  Discovery: {len([r for r in matrix.experiments if r.phase == 'discovery'])} runs")
        print(f"  Migration: {len([r for r in matrix.experiments if r.phase == 'migration'])} runs")

    # Export rule cards
    if args.export_cards:
        cards_path = os.path.join(output_dir, "rule_cards.json")
        with open(cards_path, "w") as f:
            f.write(registry.to_json())
        print(f"Rule cards ({len(registry.all_rules())} rules) saved to {cards_path}")

        # Also export as markdown
        for rule in registry.all_rules():
            md_path = os.path.join(output_dir, f"rule_card_{rule.rule_name}.md")
            with open(md_path, "w") as f:
                f.write(rule.to_markdown())
        print(f"  Individual rule card markdown files saved")

    # Run validation if labeled windows data is provided
    if args.validation_data and os.path.isfile(args.validation_data):
        with open(args.validation_data) as f:
            val_data = json.load(f)

        discovery_reports = {}
        for rule_name, rule_data in val_data.items():
            windows = [LabeledWindow(**w) for w in rule_data.get("labeled_windows", [])]
            seeds = [SeedResult(**s) for s in rule_data.get("seed_results", [])]
            intervention_data = rule_data.get("intervention_data")
            report = validate_rule(rule_name, windows, seeds, intervention_data)
            discovery_reports[rule_name] = report

            # Update rule card
            card = registry.get(rule_name)
            if card:
                from .rule_eval.rule_validator import update_rule_card
                update_rule_card(card, report)

            # Save validation report
            report_path = os.path.join(output_dir, f"validation_{rule_name}.json")
            with open(report_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            report_md_path = os.path.join(output_dir, f"validation_{rule_name}.md")
            with open(report_md_path, "w") as f:
                f.write(report.to_markdown())
            print(f"  {rule_name}: {'PASS' if report.overall_pass else 'FAIL'} "
                  f"(P={report.precision:.3f}, R={report.recall:.3f})")

        # Generate migration report if migration data is provided
        migration_data = {}
        if args.migration_data and os.path.isfile(args.migration_data):
            with open(args.migration_data) as f:
                migration_data = json.load(f)

        if migration_data:
            mig_report = generate_migration_report(
                registry, discovery_reports, migration_data,
                experiment_matrix=matrix,
                output_path=os.path.join(output_dir, "migration_report.json"),
            )
            mig_md_path = os.path.join(output_dir, "migration_report.md")
            with open(mig_md_path, "w") as f:
                f.write(mig_report.to_markdown())
            print(f"  Migration report: {'PASS' if mig_report.overall_pass else 'PENDING/FAIL'}")
            print(f"  {mig_report.summary}")

    # Save updated rule cards
    updated_cards_path = os.path.join(output_dir, "rule_cards_final.json")
    with open(updated_cards_path, "w") as f:
        f.write(registry.to_json())

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="rl_diagnostics",
        description="RL training diagnostics tool for verl",
    )
    sub = parser.add_subparsers(dest="command")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Run full offline analysis")
    p_analyze.add_argument("--log", help="Path to training log file")
    p_analyze.add_argument("--metrics", help="Path to metrics file (optional)")
    p_analyze.add_argument("--script", help="Path to start.sh (fallback)")
    p_analyze.add_argument("--rollout-dir", help="Path to rollout JSONL directory")
    p_analyze.add_argument("--dataset", help="Path to training dataset (parquet)")
    p_analyze.add_argument("--difficulty-map", help="Path to difficulty mapping JSON")
    p_analyze.add_argument("--experiment-id", help="Override experiment ID")
    p_analyze.add_argument("--config", help="Path to diagnostics config file")
    p_analyze.add_argument("--output-dir", default="./diagnostics_output", help="Output directory")
    p_analyze.add_argument("--abs-epsilon", type=float, default=1e-6, help="Absolute reward tolerance")
    p_analyze.add_argument("--rel-epsilon", type=float, default=1e-4, help="Relative reward tolerance")
    p_analyze.add_argument("--min-usable", type=int, default=2, help="Min usable rollouts per group")
    p_analyze.add_argument("--min-effective-ratio", type=float, default=0.3, help="Min informative pair ratio")
    p_analyze.add_argument("--extreme-ratio-threshold", type=float, default=0.75,
                           help="Export groups at or beyond this observed correct/reward ratio")
    p_analyze.add_argument("--filter-steps", help="Comma-separated step filter for export")
    p_analyze.add_argument("--filter-types", help="Comma-separated diagnosis type filter")
    p_analyze.add_argument("--filter-difficulties", help="Comma-separated difficulty filter")
    p_analyze.add_argument("--no-html", action="store_true", help="Skip HTML report generation")
    p_analyze.add_argument("--export-svgs", action="store_true", help="Export standalone SVG chart files")
    p_analyze.set_defaults(func=cmd_analyze)

    # validate
    p_validate = sub.add_parser("validate", help="Validate log and JSONL schema")
    p_validate.add_argument("--log", help="Path to training log file")
    p_validate.add_argument("--script", help="Path to start.sh (fallback)")
    p_validate.add_argument("--rollout-dir", help="Path to rollout JSONL directory")
    p_validate.set_defaults(func=cmd_validate)

    # export
    p_export = sub.add_parser("export", help="Export abnormal prompts")
    p_export.add_argument("--log", help="Path to training log file")
    p_export.add_argument("--rollout-dir", help="Path to rollout JSONL directory")
    p_export.add_argument("--output-dir", default="./diagnostics_output", help="Output directory")
    p_export.set_defaults(func=cmd_export)

    # report
    p_report = sub.add_parser("report", help="Generate HTML report from existing diagnostics output")
    p_report.add_argument("--output-dir", default="./diagnostics_output", help="Diagnostics output directory")
    p_report.add_argument("--html-path", default=None, help="Output HTML file path (default: <output-dir>/diagnostics_report.html)")
    p_report.add_argument("--export-svgs", action="store_true", help="Export standalone SVG chart files from existing output")
    p_report.set_defaults(func=cmd_report)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run domain evaluation (math, code, risk)")
    p_eval.add_argument("--domain", required=True, choices=["math", "code", "risk"], help="Evaluation domain")
    p_eval.add_argument("--rollout-dir", help="Path to rollout JSONL directory")
    p_eval.add_argument("--rollouts-jsonl", help="Path to normalized rollouts JSONL file")
    p_eval.add_argument("--test-cases", help="Path to test cases JSON file (for code eval)")
    p_eval.add_argument("--eval-data", help="Path to eval data JSON with labels/predictions (for risk eval)")
    p_eval.add_argument("--experiment-id", help="Override experiment ID")
    p_eval.add_argument("--dataset-type", default="auto", help="Dataset type (auto, gsm8k, math, humaneval, mbpp)")
    p_eval.add_argument("--output-dir", default="./eval_output", help="Output directory")
    p_eval.add_argument("--tolerance", type=float, default=1e-6, help="Numerical tolerance (math eval)")
    p_eval.add_argument("--pass-k", type=int, default=None, help="Pass@k value")
    p_eval.add_argument("--timeout", type=float, default=10.0, help="Sandbox timeout seconds (code eval)")
    p_eval.add_argument("--no-sandbox", action="store_true", help="Disable sandbox (code eval returns blocked)")
    p_eval.add_argument("--positive-label", default=1, help="Positive label value (risk eval)")
    p_eval.add_argument("--has-probability", action="store_true", help="Predictions are probabilities (risk eval)")
    p_eval.add_argument("--top-k", type=int, default=None, help="Top-k for recall/lift (risk eval)")
    p_eval.set_defaults(func=cmd_evaluate)

    # rule-eval
    p_rule = sub.add_parser("rule-eval", help="Rule validity and migration evaluation (REQ-12)")
    p_rule.add_argument("--output-dir", default="./rule_eval_output", help="Output directory")
    p_rule.add_argument("--show-matrix", action="store_true", help="Generate and display experiment matrix")
    p_rule.add_argument("--export-cards", action="store_true", help="Export rule cards")
    p_rule.add_argument("--validation-data", help="Path to validation data JSON (labeled windows, seeds)")
    p_rule.add_argument("--migration-data", help="Path to migration data JSON")
    p_rule.set_defaults(func=cmd_rule_eval)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
