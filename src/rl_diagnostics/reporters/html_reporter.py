"""REQ-11: Local HTML diagnostic report generator.

Produces a single self-contained HTML file with embedded SVG charts.
Works completely offline - no external JavaScript, CSS, or fonts.
"""
from __future__ import annotations

import html
import json
import os
from datetime import datetime
from typing import Any, Optional

from ..schemas.data_models import ExperimentRecord, ParseReport
from ..analyzers.group_signal import StepAggregation, GroupSignalResult
from ..analyzers.rollout_quality import RolloutQualityResult
from ..analyzers.stability_analyzer import StabilityReport
from ..analyzers.difficulty_analyzer import DifficultyReport
from .visualizer import (
    chart_step_signal, chart_epoch_comparison,
    chart_rollout_quality, chart_difficulty_stratification,
    chart_group_reward_distribution, chart_rollout_quality_timeline,
    chart_group_signal_heatmap, chart_invalid_prompt_cards,
)


def _esc(text: Any) -> str:
    """HTML-escape a value."""
    return html.escape(str(text), quote=True)


def _section(title: str, content: str, section_id: str = "") -> str:
    """Wrap content in a section."""
    sid = f' id="{section_id}"' if section_id else ""
    return f'<section{sid}><h2>{_esc(title)}</h2>{content}</section>'


def _kv_table(rows: list[tuple[str, Any]], title: str = "") -> str:
    """Render a key-value table."""
    parts = []
    if title:
        parts.append(f"<h3>{_esc(title)}</h3>")
    parts.append('<table class="kv-table"><tbody>')
    for key, val in rows:
        display = "—" if val is None or val == "" else _esc(val)
        parts.append(f"<tr><th>{_esc(key)}</th><td>{display}</td></tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _alert_card(alert: dict) -> str:
    """Render a single stability alert as a card."""
    severity_class = f"alert-{alert.get('severity', 'info')}"
    parts = [f'<div class="alert-card {severity_class}">']
    parts.append(
        f'<div class="alert-header">'
        f'<span class="alert-rule">{_esc(alert["rule_name"])}</span>'
        f'<span class="alert-severity">{_esc(alert["severity"])}</span>'
        f'<span class="alert-metric">{_esc(alert["metric_name"])}</span>'
        f"</div>"
    )
    parts.append(f'<p class="alert-desc">{_esc(alert.get("description", ""))}</p>')
    parts.append(
        f'<div class="alert-evidence">'
        f'<span>Trigger: <code>{_esc(alert.get("trigger_value", ""))}</code></span>'
        f'<span>Baseline: <code>{_esc(alert.get("baseline_value", ""))}</code></span>'
        f'<span>Threshold: <code>{_esc(alert.get("threshold", ""))}</code></span>'
        f'<span>Phase: <code>{_esc(alert.get("phase", ""))}</code></span>'
        f"</div>"
    )
    steps = alert.get("window_steps", [])
    if steps:
        parts.append(f'<div class="alert-window">Window steps: {", ".join(str(s) for s in steps[:20])}')
        if len(steps) > 20:
            parts.append(f" (+{len(steps) - 20} more)")
        parts.append("</div>")
    hints = alert.get("investigation_hints", [])
    if hints:
        parts.append('<ul class="alert-hints">')
        for h in hints:
            parts.append(f"<li>{_esc(h)}</li>")
        parts.append("</ul>")
    parts.append("</div>")
    return "".join(parts)


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #f5f5f5; color: #333; line-height: 1.6; padding: 20px; }
.container { max-width: 1200px; margin: 0 auto; }
header { background: #1a237e; color: #fff; padding: 20px 30px; border-radius: 8px; margin-bottom: 20px; }
header h1 { font-size: 24px; margin-bottom: 5px; }
header .meta { font-size: 13px; opacity: 0.8; }
section { background: #fff; border-radius: 8px; padding: 20px 30px; margin-bottom: 20px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
h2 { font-size: 20px; color: #1a237e; border-bottom: 2px solid #e8eaf6; padding-bottom: 8px; margin-bottom: 15px; }
h3 { font-size: 16px; color: #283593; margin: 15px 0 8px; }
table { width: 100%; border-collapse: collapse; margin: 10px 0; }
.kv-table th { text-align: right; width: 30%; padding: 6px 12px; background: #f5f5f5;
               font-weight: 600; color: #555; border: 1px solid #e0e0e0; }
.kv-table td { padding: 6px 12px; border: 1px solid #e0e0e0; }
.data-table th { background: #e8eaf6; padding: 8px 12px; text-align: left;
                 border: 1px solid #c5cae9; font-size: 13px; }
.data-table td { padding: 6px 12px; border: 1px solid #e0e0e0; font-size: 13px; }
.data-table tr:nth-child(even) { background: #fafafa; }
.chart-container { text-align: center; margin: 15px 0; overflow-x: auto; }
.chart-container svg { max-width: 100%; height: auto; }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }
.summary-card { background: #f5f5f5; border-radius: 6px; padding: 15px; text-align: center; }
.summary-card .value { font-size: 28px; font-weight: bold; color: #1a237e; }
.summary-card .label { font-size: 13px; color: #666; }
.alert-card { border-left: 4px solid; padding: 12px 16px; margin: 10px 0; border-radius: 4px; background: #fafafa; }
.alert-critical { border-color: #f44336; background: #ffebee; }
.alert-warning { border-color: #ff9800; background: #fff3e0; }
.alert-info { border-color: #2196f3; background: #e3f2fd; }
.alert-header { display: flex; gap: 15px; align-items: center; margin-bottom: 8px; }
.alert-rule { font-weight: bold; font-size: 15px; }
.alert-severity { font-size: 12px; padding: 2px 8px; border-radius: 12px; color: #fff; }
.alert-critical .alert-severity { background: #f44336; }
.alert-warning .alert-severity { background: #ff9800; }
.alert-info .alert-severity { background: #2196f3; }
.alert-metric { font-family: monospace; font-size: 13px; color: #666; }
.alert-desc { margin: 8px 0; font-size: 14px; }
.alert-evidence { display: flex; gap: 20px; flex-wrap: wrap; font-size: 13px; margin: 8px 0; }
.alert-evidence code { background: #e0e0e0; padding: 1px 6px; border-radius: 3px; font-size: 12px; }
.alert-window { font-size: 13px; color: #666; margin: 5px 0; }
.alert-hints { margin: 8px 0 0 20px; font-size: 13px; }
.alert-hints li { margin: 2px 0; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; }
.badge-ok { background: #c8e6c9; color: #2e7d32; }
.badge-warn { background: #fff3e0; color: #e65100; }
.badge-fail { background: #ffebee; color: #c62828; }
.badge-skip { background: #f5f5f5; color: #999; }
.nav { margin-bottom: 20px; }
.nav a { display: inline-block; padding: 4px 12px; margin: 2px; background: #e8eaf6;
         color: #1a237e; text-decoration: none; border-radius: 4px; font-size: 13px; }
.nav a:hover { background: #c5cae9; }
.degraded { background: #fff3e0; padding: 10px 15px; border-radius: 4px; margin: 10px 0; font-size: 14px; }
code { font-family: "Fira Code", "Consolas", monospace; }
.footer { text-align: center; padding: 20px; color: #999; font-size: 12px; }
"""


def generate_html_report(
    experiment: Optional[ExperimentRecord] = None,
    parse_report: Optional[ParseReport] = None,
    step_aggs: Optional[list[StepAggregation]] = None,
    quality_results: Optional[list[RolloutQualityResult]] = None,
    group_signals: Optional[list[GroupSignalResult]] = None,
    stability_report: Optional[StabilityReport] = None,
    diff_report: Optional[DifficultyReport] = None,
    summary: Optional[dict] = None,
    output_path: str = "diagnostics_report.html",
    invalid_prompts_path: Optional[str] = None,
    eval_results: Optional[dict] = None,
) -> str:
    """Generate a self-contained HTML diagnostic report.

    Returns the path to the generated HTML file.
    """
    step_aggs = step_aggs or []
    quality_results = quality_results or []
    group_signals = group_signals or []
    summary = summary or {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parts: list[str] = []

    # ── Head ─────────────────────────────────────────────────────
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="zh"><head><meta charset="UTF-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
    parts.append(f"<title>RL Diagnostics Report</title>")
    parts.append(f"<style>{_CSS}</style>")
    parts.append("</head><body><div class='container'>")

    # ── Header ───────────────────────────────────────────────────
    exp_id = experiment.experiment_id if experiment else summary.get("experiment_id", "unknown")
    parts.append("<header>")
    parts.append(f"<h1>RL Training Diagnostics Report</h1>")
    parts.append(f'<div class="meta">Experiment: {_esc(exp_id)} | Generated: {now}</div>')
    parts.append("</header>")

    # ── Nav ──────────────────────────────────────────────────────
    parts.append('<nav class="nav">')
    for href, label in [
        ("#overview", "Overview"),
        ("#config", "Config"),
        ("#signal", "Learning Signal"),
        ("#epoch", "Epoch Comparison"),
        ("#quality", "Data Quality"),
        ("#difficulty", "Difficulty"),
        ("#stability", "Stability"),
        ("#prompts", "Invalid Prompts"),
        ("#rules", "Rules &amp; Config"),
    ]:
        parts.append(f'<a href="{href}">{label}</a>')
    parts.append("</nav>")

    # ── Overview / Summary ───────────────────────────────────────
    overview_content = ['<div class="summary-grid">']
    cards = [
        ("Total Rollouts", summary.get("total_rollouts", len(quality_results))),
        ("Usable Rollouts", summary.get("usable_rollouts",
            sum(1 for q in quality_results if q.status == "usable"))),
        ("Total Groups", summary.get("total_groups", len(group_signals))),
        ("Effective Groups", summary.get("effective_groups",
            sum(1 for gs in group_signals if gs.signal_type == "effective_group"))),
        ("Low-Info Groups", summary.get("low_info_groups",
            sum(1 for gs in group_signals if gs.low_information))),
        ("Exported Prompts", summary.get("exported_prompts", 0)),
    ]
    if stability_report:
        cards.append(("Stability Alerts", stability_report.total_alerts))
        cards.append(("Critical Alerts", stability_report.critical_count))
    for label, val in cards:
        overview_content.append(
            f'<div class="summary-card"><div class="value">{_esc(val)}</div>'
            f'<div class="label">{_esc(label)}</div></div>'
        )
    overview_content.append("</div>")

    # Data integrity badge
    if parse_report:
        if parse_report.failed_lines == 0:
            badge = '<span class="badge badge-ok">PASS</span>'
        else:
            badge = f'<span class="badge badge-fail">{parse_report.failed_lines} failures</span>'
        overview_content.append(
            f'<p>Data Integrity: {badge} '
            f'({parse_report.parsed_lines} parsed, {parse_report.failed_lines} failed)</p>'
        )
    parts.append(_section("Overview", "".join(overview_content), "overview"))

    # ── Experiment Config ────────────────────────────────────────
    if experiment:
        config_rows = [
            ("Experiment ID", experiment.experiment_id),
            ("Experiment Name", experiment.experiment_name),
            ("Model", experiment.model),
            ("Algorithm", experiment.algorithm),
            ("Dataset", experiment.dataset),
            ("Rollout N", experiment.rollout_n),
            ("Train Batch Size", experiment.train_batch_size),
            ("Learning Rate", experiment.learning_rate),
            ("Total Epochs", experiment.total_epochs),
            ("Total Steps", experiment.total_training_steps),
            ("Schema Version", experiment.schema_version),
        ]
        parts.append(_section("Experiment Configuration",
                              _kv_table(config_rows), "config"))

        if experiment.config_conflicts:
            conflict_rows = "".join(
                f"<tr><td>{_esc(c.get('key',''))}</td>"
                f"<td>{_esc(c.get('values',''))}</td>"
                f"<td>{_esc(c.get('resolved',''))}</td></tr>"
                for c in experiment.config_conflicts
            )
            parts.append(_section("Config Conflicts",
                f'<table class="data-table"><thead><tr>'
                f'<th>Key</th><th>Values</th><th>Resolved</th></tr></thead>'
                f'<tbody>{conflict_rows}</tbody></table>'))

    # ── Learning Signal Chart ────────────────────────────────────
    signal_svg = chart_step_signal(step_aggs)
    parts.append(_section("Per-Step Learning Signal",
        f'<div class="chart-container">{signal_svg}</div>'
        '<p>Stacked bars show group classification per step. '
        'Blue line shows effective group ratio.</p>',
        "signal"))

    # ── Group Reward Distribution ───────────────────────────────
    if group_signals:
        reward_svg = chart_group_reward_distribution(group_signals)
        parts.append(_section("Group Reward Distribution",
            f'<div class="chart-container">{reward_svg}</div>'
            '<p>Each row is a group; dots show individual rollout rewards. '
            'Color indicates signal type (green=effective, red=constant).</p>',
            "reward-distribution"))

    # ── Rollout Quality Timeline ────────────────────────────────
    if quality_results:
        timeline_svg = chart_rollout_quality_timeline(quality_results)
        parts.append(_section("Rollout Quality Timeline",
            f'<div class="chart-container">{timeline_svg}</div>'
            '<p>Stacked bars show rollout quality composition per step.</p>',
            "quality-timeline"))

    # ── Group Signal Heatmap ────────────────────────────────────
    if group_signals:
        heatmap_svg = chart_group_signal_heatmap(group_signals)
        parts.append(_section("Group Signal Heatmap",
            f'<div class="chart-container">{heatmap_svg}</div>'
            '<p>Matrix view: rows=steps, columns=groups, color=signal type. '
            'Numbers show informative pair ratio (IPR).</p>',
            "signal-heatmap"))

    # ── Epoch Comparison ─────────────────────────────────────────
    epoch_svg = chart_epoch_comparison(step_aggs)
    parts.append(_section("Epoch Comparison",
        f'<div class="chart-container">{epoch_svg}</div>', "epoch"))

    # ── Data Quality ─────────────────────────────────────────────
    quality_svg = chart_rollout_quality(quality_results)
    parts.append(_section("Rollout Data Quality",
        f'<div class="chart-container">{quality_svg}</div>', "quality"))

    # ── Difficulty ───────────────────────────────────────────────
    if diff_report:
        diff_svg = chart_difficulty_stratification(diff_report)
        diff_content = f'<div class="chart-container">{diff_svg}</div>'
        if not diff_report.has_difficulty:
            diff_content += (
                '<div class="degraded">Difficulty field not provided in data. '
                'No automatic inference is performed. Stratification is skipped.</div>'
            )
        else:
            diff_content += f'<p>Missing rate: {diff_report.missing_rate:.1%}</p>'
        parts.append(_section("Difficulty Stratification", diff_content, "difficulty"))

    # ── Stability ────────────────────────────────────────────────
    if stability_report:
        stab_parts = []
        stab_parts.append(
            f'<p>Rules checked: {", ".join(stability_report.rules_checked)} | '
            f'Skipped: {", ".join(stability_report.rules_skipped) or "None"}</p>'
        )
        if stability_report.alerts:
            stab_parts.append(f'<p><strong>{stability_report.total_alerts}</strong> alerts: '
                              f'{stability_report.critical_count} critical, '
                              f'{stability_report.warning_count} warning, '
                              f'{stability_report.info_count} info</p>')
            for alert in stability_report.alerts:
                stab_parts.append(_alert_card(alert.to_dict()))
        else:
            stab_parts.append(
                '<p><span class="badge badge-ok">No alerts</span> '
                'No stability issues detected with current thresholds.</p>'
            )
        parts.append(_section("Training Stability", "".join(stab_parts), "stability"))

    # ── Invalid Prompts ──────────────────────────────────────────
    prompt_parts = []
    if invalid_prompts_path and os.path.isfile(invalid_prompts_path):
        with open(invalid_prompts_path, "r") as f:
            line_count = sum(1 for _ in f)
        prompt_parts.append(
            f'<p>Exported prompts: <strong>{line_count}</strong> records</p>'
        )
        prompt_parts.append(
            f'<p>File: <code>{_esc(invalid_prompts_path)}</code></p>'
        )
        # Show first 5 entries in a table
        prompt_parts.append('<table class="data-table"><thead><tr>'
                            '<th>Step</th><th>Group ID</th><th>Triggers</th>'
                            '<th>Prompt (truncated)</th></tr></thead><tbody>')
        with open(invalid_prompts_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                try:
                    d = json.loads(line)
                    prompt_text = (d.get("prompt") or "")[:100]
                    triggers = ", ".join(d.get("triggered_reasons", []))
                    prompt_parts.append(
                        f"<tr><td>{_esc(d.get('step',''))}</td>"
                        f"<td><code>{_esc(d.get('group_id',''))}</code></td>"
                        f"<td>{_esc(triggers)}</td>"
                        f"<td>{_esc(prompt_text)}...</td></tr>"
                    )
                except (json.JSONDecodeError, KeyError):
                    pass
        prompt_parts.append("</tbody></table>")
        if line_count > 5:
            prompt_parts.append(f"<p>... and {line_count - 5} more records in the export file.</p>")
        # Abnormal prompt card visualization
        invalid_prompt_data: list[dict] = []
        with open(invalid_prompts_path, "r") as f:
            for line in f:
                try:
                    invalid_prompt_data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if invalid_prompt_data:
            cards_svg = chart_invalid_prompt_cards(invalid_prompt_data)
            prompt_parts.append(f'<div class="chart-container">{cards_svg}</div>')
    else:
        prompt_parts.append(
            '<div class="degraded">No invalid prompts export file found.</div>'
        )
    # ── Domain Evaluation Results ─────────────────────────────────
    if eval_results:
        eval_parts: list[str] = []
        for domain, result in eval_results.items():
            eval_parts.append(f'<h3>{domain.title()} Evaluation</h3>')
            if isinstance(result, dict):
                if result.get("degenerate"):
                    eval_parts.append(
                        f'<div class="alert alert-warning">Degenerate: '
                        f'{_esc(result.get("degenerate_reason", ""))}</div>'
                    )
                if result.get("blocked"):
                    eval_parts.append(
                        f'<div class="alert alert-warning">Blocked: '
                        f'{_esc(result.get("blocked_reason", ""))}</div>'
                    )
                # Key metrics
                metrics_html = []
                metric_labels = {
                    "parse_success_rate": "Parse Success Rate",
                    "exact_match": "Exact Match",
                    "equivalent_accuracy": "Equivalent Accuracy",
                    "pass_at_1": "Pass@1",
                    "pass_at_k": "Pass@k",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1": "F1",
                    "roc_auc": "ROC-AUC",
                    "pr_auc": "PR-AUC",
                    "ks": "KS",
                    "extraction_success_rate": "Extraction Success Rate",
                    "compile_success_rate": "Compile Success Rate",
                    "test_pass_rate": "Test Pass Rate",
                    "timeout_rate": "Timeout Rate",
                    "exception_rate": "Exception Rate",
                    "unsafe_api_rate": "Unsafe API Rate",
                    "reward_eval_agreement": "Reward-Eval Agreement",
                }
                for key, label in metric_labels.items():
                    val = result.get(key)
                    if val is not None and not isinstance(val, (list, dict)):
                        metrics_html.append(
                            f'<span class="badge badge-ok">{label}: {val:.3f}</span>'
                        )
                if metrics_html:
                    eval_parts.append('<div class="metrics-row">' + " ".join(metrics_html) + '</div>')

                # Badcase counts
                badcases = result.get("badcase_counts", {})
                if badcases:
                    bc_items = " ".join(
                        f'<span class="badge badge-warn">{_esc(k)}: {v}</span>'
                        for k, v in badcases.items()
                    )
                    eval_parts.append(f'<div class="metrics-row">Badcases: {bc_items}</div>')

                # Reward-eval disagreements
                disagree = result.get("reward_eval_disagreement_count", 0)
                if disagree and disagree > 0:
                    eval_parts.append(
                        f'<div class="alert alert-warning">Reward-Eval disagreements: {disagree} samples</div>'
                    )
        if eval_parts:
            parts.append(_section("Domain Evaluation", "".join(eval_parts), "evaluation"))

    parts.append(_section("Invalid Prompt Export", "".join(prompt_parts), "prompts"))

    # ── Rules & Config ───────────────────────────────────────────
    rules_parts = []
    if stability_report:
        rules_parts.append(f"<h3>Stability Rule Config (v{_esc(stability_report.config_version)})</h3>")
        rules_parts.append('<p>Thresholds are first-round experimental starting points. '
                           'They must be calibrated via REQ-12 before production use.</p>')
        rules_parts.append(
            f'<p>Available metrics: {", ".join(_esc(m) for m in stability_report.metrics_available[:20])}'
            + ("..." if len(stability_report.metrics_available) > 20 else "") + "</p>"
        )

    # Unevaluated items
    unevaluated = []
    for req in ("REQ-08", "REQ-09", "REQ-10", "REQ-12"):
        unevaluated.append(f'<span class="badge badge-skip">{req}</span>')
    rules_parts.append(
        '<h3>Unevaluated Diagnostics</h3>'
        '<p>The following requirements are not yet implemented:</p>'
        + " ".join(unevaluated)
    )
    parts.append(_section("Rules, Thresholds & Unevaluated", "".join(rules_parts), "rules"))

    # ── Footer ───────────────────────────────────────────────────
    parts.append(
        '<div class="footer">'
        f'Generated by RL Diagnostics Tool | Schema v1.0.0 | {now}'
        '</div>'
    )

    parts.append("</div></body></html>")

    # Write
    html_content = "\n".join(parts)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path
