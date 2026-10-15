"""REQ-04: Rollout visualization using pure-Python SVG generation.

Generates embedded SVG charts that work in offline HTML reports.
No external JavaScript or CSS dependencies required.
"""
from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Any, Optional

from ..analyzers.group_signal import StepAggregation
from ..analyzers.rollout_quality import RolloutQualityResult
from ..analyzers.difficulty_analyzer import DifficultyReport


@dataclass
class ChartTheme:
    """Color palette for charts."""
    bg: str = "#ffffff"
    text: str = "#333333"
    grid: str = "#e0e0e0"
    axis: str = "#999999"
    effective: str = "#4caf50"
    constant: str = "#f44336"
    near_constant: str = "#ff9800"
    insufficient: str = "#9e9e9e"
    unresolved: str = "#607d8b"
    line_primary: str = "#2196f3"
    line_secondary: str = "#9c27b0"
    quality_empty: str = "#f44336"
    quality_reward: str = "#ff5722"
    quality_format: str = "#ff9800"
    quality_truncated: str = "#9c27b0"
    quality_verifier: str = "#795548"


def _escape(text: str) -> str:
    """HTML-escape text for SVG output."""
    return html.escape(str(text), quote=True)


def _fmt(val: float, decimals: int = 2) -> str:
    """Format a float for display."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


# ── SVG primitives ───────────────────────────────────────────────

class SVGChart:
    """Builder for an SVG chart."""

    def __init__(self, width: int = 800, height: int = 400, theme: Optional[ChartTheme] = None):
        self.width = width
        self.height = height
        self.theme = theme or ChartTheme()
        self.elements: list[str] = []

    def rect(self, x: float, y: float, w: float, h: float,
             fill: str = "none", stroke: str = "none", sw: float = 0,
             opacity: float = 1.0, rx: float = 0) -> None:
        attrs = f'x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"'
        attrs += f' fill="{fill}"'
        if stroke != "none":
            attrs += f' stroke="{stroke}" stroke-width="{sw:.1f}"'
        if opacity < 1.0:
            attrs += f' opacity="{opacity}"'
        if rx > 0:
            attrs += f' rx="{rx:.1f}"'
        self.elements.append(f'<rect {attrs}/>')

    def line(self, x1: float, y1: float, x2: float, y2: float,
             stroke: str = "#333", sw: float = 1, dash: str = "") -> None:
        attrs = f'x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
        attrs += f' stroke="{stroke}" stroke-width="{sw:.1f}"'
        if dash:
            attrs += f' stroke-dasharray="{dash}"'
        self.elements.append(f'<line {attrs}/>')

    def text(self, x: float, y: float, content: str,
             size: int = 12, fill: str = "#333", anchor: str = "start",
             weight: str = "normal", rotate: float = 0) -> None:
        attrs = f'x="{x:.1f}" y="{y:.1f}" font-size="{size}"'
        attrs += f' fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"'
        attrs += ' font-family="sans-serif"'
        if rotate:
            attrs += f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"'
        self.elements.append(f'<text {attrs}>{_escape(content)}</text>')

    def polyline(self, points: list[tuple[float, float]],
                 stroke: str = "#333", sw: float = 2, fill: str = "none") -> None:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        attrs = f'points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}"'
        self.elements.append(f'<polyline {attrs}/>')

    def circle(self, cx: float, cy: float, r: float,
               fill: str = "#333", stroke: str = "none", sw: float = 0) -> None:
        attrs = f'cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"'
        if stroke != "none":
            attrs += f' stroke="{stroke}" stroke-width="{sw:.1f}"'
        self.elements.append(f'<circle {attrs}/>')

    def title(self, text: str, size: int = 16) -> None:
        self.text(self.width / 2, 25, text, size=size, fill=self.theme.text,
                  anchor="middle", weight="bold")

    def render(self) -> str:
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}"'
        svg += f' viewBox="0 0 {self.width} {self.height}">\n'
        svg += f'<rect width="{self.width}" height="{self.height}" fill="{self.theme.bg}"/>\n'
        svg += "\n".join(self.elements)
        svg += "\n</svg>"
        return svg


# ── Chart builders ───────────────────────────────────────────────

def chart_step_signal(
    step_aggs: list[StepAggregation],
    theme: Optional[ChartTheme] = None,
    width: int = 900,
    height: int = 400,
) -> str:
    """REQ-04 Chart 1: Per-step learning signal (stacked bar + ratio line)."""
    theme = theme or ChartTheme()
    chart = SVGChart(width, height, theme)
    chart.title("Per-Step Learning Signal")

    if not step_aggs:
        chart.text(width / 2, height / 2, "No step data available", size=14,
                   fill=theme.text, anchor="middle")
        return chart.render()

    margin_l, margin_r, margin_t, margin_b = 60, 60, 40, 60
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    n = len(step_aggs)
    bar_w = max(1, plot_w / n - 2)
    max_groups = max(sa.analyzable_groups for sa in step_aggs) or 1

    # Y axis
    chart.line(margin_l, margin_t, margin_l, margin_t + plot_h, stroke=theme.axis, sw=1)
    chart.line(margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h,
               stroke=theme.axis, sw=1)
    chart.text(margin_l - 10, margin_t + plot_h / 2, "Groups", size=11,
               fill=theme.text, anchor="end", rotate=-90)

    # Grid lines
    for frac in (0.25, 0.50, 0.75, 1.0):
        y = margin_t + plot_h * (1 - frac)
        chart.line(margin_l, y, margin_l + plot_w, y, stroke=theme.grid, sw=0.5, dash="4,4")
        chart.text(margin_l - 5, y + 4, str(int(max_groups * frac)), size=10,
                   fill=theme.axis, anchor="end")

    # Stacked bars
    for i, sa in enumerate(step_aggs):
        x = margin_l + i * (plot_w / n)
        y_bottom = margin_t + plot_h
        bar_x = x + (plot_w / n - bar_w) / 2

        segments = [
            (sa.effective_groups, theme.effective, "effective"),
            (sa.constant_score_groups, theme.constant, "constant"),
            (sa.near_constant_groups, theme.near_constant, "near_const"),
            (sa.insufficient_groups, theme.insufficient, "insufficient"),
            (sa.unresolved_groups, theme.unresolved, "unresolved"),
        ]

        for count, color, _label in segments:
            if count <= 0:
                continue
            seg_h = (count / max_groups) * plot_h
            chart.rect(bar_x, y_bottom - seg_h, bar_w, seg_h, fill=color)
            y_bottom -= seg_h

    # Effective ratio line (secondary axis 0-1)
    line_points = []
    for i, sa in enumerate(step_aggs):
        x = margin_l + i * (plot_w / n) + (plot_w / n) / 2
        y = margin_t + plot_h * (1 - sa.effective_group_ratio)
        line_points.append((x, y))
    if line_points:
        chart.polyline(line_points, stroke=theme.line_primary, sw=2)
        for x, y in line_points:
            chart.circle(x, y, 2.5, fill=theme.line_primary)

    # X axis labels (show every Nth step)
    label_interval = max(1, n // 15)
    for i, sa in enumerate(step_aggs):
        if i % label_interval != 0 and i != n - 1:
            continue
        x = margin_l + i * (plot_w / n) + (plot_w / n) / 2
        chart.text(x, margin_t + plot_h + 20, str(sa.step), size=10,
                   fill=theme.axis, anchor="middle")
    chart.text(width / 2, height - 10, "Step", size=11, fill=theme.text, anchor="middle")

    # Right Y axis for ratio
    chart.line(margin_l + plot_w, margin_t, margin_l + plot_w, margin_t + plot_h,
               stroke=theme.axis, sw=1)
    chart.text(margin_l + plot_w + 15, margin_t + plot_h / 2, "Effective Ratio",
               size=11, fill=theme.line_primary, anchor="start", rotate=90)
    for frac in (0.0, 0.5, 1.0):
        y = margin_t + plot_h * (1 - frac)
        chart.text(margin_l + plot_w + 5, y + 4, f"{frac:.1f}", size=10,
                   fill=theme.line_primary, anchor="start")

    # Legend
    legend_items = [
        ("Effective", theme.effective),
        ("Constant", theme.constant),
        ("Near-Const", theme.near_constant),
        ("Insufficient", theme.insufficient),
        ("Unresolved", theme.unresolved),
        ("Eff. Ratio", theme.line_primary),
    ]
    lx = margin_l
    ly = height - 22
    for label, color in legend_items:
        chart.rect(lx, ly - 8, 12, 10, fill=color)
        chart.text(lx + 16, ly, label, size=9, fill=theme.text)
        lx += 80

    return chart.render()


def chart_epoch_comparison(
    step_aggs: list[StepAggregation],
    theme: Optional[ChartTheme] = None,
    width: int = 700,
    height: int = 350,
) -> str:
    """REQ-04 Chart 2: Epoch comparison (grouped bar)."""
    theme = theme or ChartTheme()
    chart = SVGChart(width, height, theme)
    chart.title("Epoch Comparison")

    # Group by epoch
    epochs: dict[int, list[StepAggregation]] = {}
    for sa in step_aggs:
        if sa.epoch is not None:
            epochs.setdefault(sa.epoch, []).append(sa)

    if not epochs:
        chart.text(width / 2, height / 2, "No epoch data available", size=14,
                   fill=theme.text, anchor="middle")
        return chart.render()

    sorted_epochs = sorted(epochs.keys())
    margin_l, margin_r, margin_t, margin_b = 60, 30, 40, 60
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    metrics = [
        ("Effective Ratio", "effective_group_ratio", theme.effective),
        ("Low-Info Ratio", "low_information_group_ratio", theme.constant),
        ("Mean IPR", "mean_informative_pair_ratio", theme.line_primary),
    ]

    n_epochs = len(sorted_epochs)
    n_metrics = len(metrics)
    group_w = plot_w / n_epochs
    bar_w = (group_w - 10) / n_metrics

    # Axes
    chart.line(margin_l, margin_t, margin_l, margin_t + plot_h, stroke=theme.axis, sw=1)
    chart.line(margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h,
               stroke=theme.axis, sw=1)

    # Grid
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        y = margin_t + plot_h * (1 - frac)
        chart.line(margin_l, y, margin_l + plot_w, y, stroke=theme.grid, sw=0.5, dash="4,4")
        chart.text(margin_l - 5, y + 4, f"{frac:.1f}", size=10, fill=theme.axis, anchor="end")

    for ei, epoch in enumerate(sorted_epochs):
        agg_list = epochs[epoch]
        x_base = margin_l + ei * group_w + 5

        # Average metrics across steps in this epoch
        for mi, (_label, attr, color) in enumerate(metrics):
            vals = [getattr(sa, attr, 0.0) for sa in agg_list]
            avg = sum(vals) / len(vals) if vals else 0.0
            bar_h = avg * plot_h
            chart.rect(x_base + mi * bar_w, margin_t + plot_h - bar_h, bar_w - 2, bar_h,
                       fill=color)

        chart.text(x_base + group_w / 2 - 5, margin_t + plot_h + 20,
                   f"Epoch {epoch}", size=11, fill=theme.text, anchor="middle")

    chart.text(margin_l - 10, margin_t + plot_h / 2, "Ratio", size=11,
               fill=theme.text, anchor="end", rotate=-90)

    # Legend
    lx = margin_l
    ly = height - 18
    for label, _attr, color in metrics:
        chart.rect(lx, ly - 8, 12, 10, fill=color)
        chart.text(lx + 16, ly, label, size=9, fill=theme.text)
        lx += 120

    return chart.render()


def chart_rollout_quality(
    quality_results: list[RolloutQualityResult],
    theme: Optional[ChartTheme] = None,
    width: int = 700,
    height: int = 350,
) -> str:
    """REQ-04 Chart 3: Rollout data quality distribution (pie/bar)."""
    theme = theme or ChartTheme()
    chart = SVGChart(width, height, theme)
    chart.title("Rollout Data Quality")

    # Count by status
    status_colors = {
        "usable": theme.effective,
        "invalid_empty": theme.quality_empty,
        "invalid_reward": theme.quality_reward,
        "invalid_format": theme.quality_format,
        "invalid_truncated": theme.quality_truncated,
        "invalid_verifier": theme.quality_verifier,
    }
    status_labels = {
        "usable": "Usable",
        "invalid_empty": "Empty",
        "invalid_reward": "Reward Missing",
        "invalid_format": "Format Error",
        "invalid_truncated": "Truncated",
        "invalid_verifier": "Verifier Error",
    }

    counts: dict[str, int] = {}
    for q in quality_results:
        counts[q.status] = counts.get(q.status, 0) + 1

    if not counts:
        chart.text(width / 2, height / 2, "No quality data available", size=14,
                   fill=theme.text, anchor="middle")
        return chart.render()

    total = sum(counts.values())
    margin_l, margin_b, margin_t = 60, 60, 50
    plot_w = width - margin_l - 20
    plot_h = height - margin_t - margin_b

    # Horizontal bar chart
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    max_count = max(c for _, c in sorted_items) or 1
    bar_h = min(25, plot_h / len(sorted_items) - 4)

    chart.line(margin_l, margin_t, margin_l, margin_t + plot_h, stroke=theme.axis, sw=1)

    for i, (status, count) in enumerate(sorted_items):
        y = margin_t + i * (bar_h + 4)
        color = status_colors.get(status, "#cccccc")
        label = status_labels.get(status, status)
        bar_w_val = (count / max_count) * (plot_w - 120)

        chart.text(margin_l - 5, y + bar_h / 2 + 4, label, size=10,
                   fill=theme.text, anchor="end")
        chart.rect(margin_l, y, bar_w_val, bar_h, fill=color)
        chart.text(margin_l + bar_w_val + 5, y + bar_h / 2 + 4,
                   f"{count} ({count / total:.1%})", size=10, fill=theme.text)

    chart.text(width / 2, height - 10, f"Total: {total} rollouts", size=11,
               fill=theme.text, anchor="middle")

    return chart.render()


def chart_difficulty_stratification(
    diff_report: DifficultyReport,
    theme: Optional[ChartTheme] = None,
    width: int = 700,
    height: int = 350,
) -> str:
    """REQ-04 Chart 4: Difficulty stratification (conditional on difficulty availability)."""
    theme = theme or ChartTheme()
    chart = SVGChart(width, height, theme)
    chart.title("Difficulty Stratification")

    if not diff_report.has_difficulty:
        chart.text(width / 2, height / 2, "Difficulty field not provided", size=14,
                   fill=theme.text, anchor="middle")
        chart.text(width / 2, height / 2 + 25,
                   "No automatic difficulty inference is performed", size=11,
                   fill=theme.axis, anchor="middle")
        return chart.render()

    buckets = diff_report.buckets
    if not buckets:
        chart.text(width / 2, height / 2, "No difficulty buckets available", size=14,
                   fill=theme.text, anchor="middle")
        return chart.render()

    margin_l, margin_r, margin_t, margin_b = 60, 30, 40, 60
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    # buckets is dict[str, DifficultyStats] - sort by key
    sorted_buckets = sorted(buckets.values(), key=lambda b: b.difficulty)
    n = len(sorted_buckets)
    bar_w = plot_w / n - 10

    # Axes
    chart.line(margin_l, margin_t, margin_l, margin_t + plot_h, stroke=theme.axis, sw=1)
    chart.line(margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h,
               stroke=theme.axis, sw=1)

    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        y = margin_t + plot_h * (1 - frac)
        chart.line(margin_l, y, margin_l + plot_w, y, stroke=theme.grid, sw=0.5, dash="4,4")
        chart.text(margin_l - 5, y + 4, f"{frac:.1f}", size=10, fill=theme.axis, anchor="end")

    for i, bucket in enumerate(sorted_buckets):
        x = margin_l + i * (plot_w / n) + 5
        eff_h = bucket.effective_group_ratio * plot_h
        const_h = bucket.constant_score_group_ratio * plot_h
        near_h = bucket.near_constant_group_ratio * plot_h

        y_bottom = margin_t + plot_h
        if eff_h > 0:
            chart.rect(x, y_bottom - eff_h, bar_w, eff_h, fill=theme.effective)
            y_bottom -= eff_h
        if const_h > 0:
            chart.rect(x, y_bottom - const_h, bar_w, const_h, fill=theme.constant)
            y_bottom -= const_h
        if near_h > 0:
            chart.rect(x, y_bottom - near_h, bar_w, near_h, fill=theme.near_constant)

        chart.text(x + bar_w / 2, margin_t + plot_h + 20,
                   _escape(bucket.difficulty), size=10,
                   fill=theme.text, anchor="middle")

    chart.text(margin_l - 10, margin_t + plot_h / 2, "Ratio", size=11,
               fill=theme.text, anchor="end", rotate=-90)

    # Legend
    lx = margin_l
    ly = height - 18
    for label, color in [("Effective", theme.effective),
                         ("Constant", theme.constant),
                         ("Near-Const", theme.near_constant)]:
        chart.rect(lx, ly - 8, 12, 10, fill=color)
        chart.text(lx + 16, ly, label, size=9, fill=theme.text)
        lx += 100

    return chart.render()


# ── Intermediate-process charts ─────────────────────────────────

def chart_group_reward_distribution(
    group_signals: list[GroupSignalResult],
    theme: Optional[ChartTheme] = None,
    width: int = 900,
    height: int = 450,
) -> str:
    """Chart 5: Per-group reward distribution.

    Shows each group as a row of reward dots, colored by signal type.
    Gives an immediate visual of which groups have reward spread
    (effective) vs flat rewards (constant).
    """
    theme = theme or ChartTheme()
    chart = SVGChart(width, height, theme)
    chart.title("Group Reward Distribution")

    if not group_signals:
        chart.text(width / 2, height / 2, "No group data available", size=14,
                   fill=theme.text, anchor="middle")
        return chart.render()

    signal_colors = {
        "effective_group": theme.effective,
        "constant_score": theme.constant,
        "near_constant_score": theme.near_constant,
        "insufficient_group": theme.insufficient,
        "group_unresolved": theme.unresolved,
    }

    sorted_groups = sorted(group_signals, key=lambda g: (g.step, g.group_id))
    n = len(sorted_groups)

    margin_l, margin_r, margin_t, margin_b = 120, 30, 40, 80
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    row_h = min(28, plot_h / max(n, 1) - 2)
    dot_r = min(5, row_h / 4)

    # Determine reward range across all groups
    all_rewards: list[float] = []
    for g in sorted_groups:
        for r in g.rewards:
            if r is not None:
                all_rewards.append(float(r))

    if not all_rewards:
        chart.text(width / 2, height / 2, "No reward values available", size=14,
                   fill=theme.text, anchor="middle")
        return chart.render()

    r_min, r_max = min(all_rewards), max(all_rewards)
    if r_max == r_min:
        r_max = r_min + 1.0
    r_range = r_max - r_min

    # Axes
    chart.line(margin_l, margin_t, margin_l, margin_t + plot_h, stroke=theme.axis, sw=1)
    chart.line(margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h,
               stroke=theme.axis, sw=1)

    # X-axis ticks (reward values)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = margin_l + frac * plot_w
        chart.line(x, margin_t, x, margin_t + plot_h, stroke=theme.grid, sw=0.5, dash="4,4")
        val = r_min + frac * r_range
        chart.text(x, margin_t + plot_h + 15, _fmt(val, 2), size=9,
                   fill=theme.axis, anchor="middle")

    # Each group is a row
    for i, g in enumerate(sorted_groups):
        y = margin_t + i * (row_h + 2) + row_h / 2
        color = signal_colors.get(g.signal_type, "#cccccc")

        # Group label
        label = g.group_id
        if len(label) > 16:
            label = label[:14] + ".."
        chart.text(margin_l - 5, y + 4, f"s{g.step} {label}", size=9,
                   fill=theme.text, anchor="end")

        # Signal type indicator bar (left edge)
        chart.rect(margin_l - 3, y - row_h / 2 + 1, 3, row_h - 2, fill=color)

        # Reward dots
        rewards = [float(r) for r in g.rewards if r is not None]
        for j, r in enumerate(rewards):
            x = margin_l + (r - r_min) / r_range * plot_w
            chart.circle(x, y, dot_r, fill=color, stroke="white", sw=0.5)

    # X-axis label
    chart.text(margin_l + plot_w / 2, height - 8, "Reward Value", size=11,
               fill=theme.text, anchor="middle")

    # Legend
    lx, ly = margin_l, height - 28
    legend_items = [
        ("Effective", theme.effective),
        ("Constant", theme.constant),
        ("Near-Const", theme.near_constant),
        ("Insufficient", theme.insufficient),
    ]
    for label, color in legend_items:
        chart.circle(lx + 4, ly, 4, fill=color)
        chart.text(lx + 12, ly + 4, label, size=9, fill=theme.text)
        lx += 90

    return chart.render()


def chart_rollout_quality_timeline(
    quality_results: list[RolloutQualityResult],
    theme: Optional[ChartTheme] = None,
    width: int = 900,
    height: int = 400,
) -> str:
    """Chart 6: Per-step rollout quality timeline (stacked bar).

    Shows how rollout quality (usable / invalid_reward / invalid_empty / ...)
    evolves across training steps. More informative than the aggregate pie
    chart because it reveals temporal degradation.
    """
    theme = theme or ChartTheme()
    chart = SVGChart(width, height, theme)
    chart.title("Rollout Quality Timeline")

    if not quality_results:
        chart.text(width / 2, height / 2, "No quality data available", size=14,
                   fill=theme.text, anchor="middle")
        return chart.render()

    status_colors = {
        "usable": theme.effective,
        "invalid_empty": theme.quality_empty,
        "invalid_reward": theme.quality_reward,
        "invalid_format": theme.quality_format,
        "invalid_truncated": theme.quality_truncated,
        "invalid_verifier": theme.quality_verifier,
    }
    status_labels = {
        "usable": "Usable",
        "invalid_empty": "Empty",
        "invalid_reward": "Reward Missing",
        "invalid_format": "Format Error",
        "invalid_truncated": "Truncated",
        "invalid_verifier": "Verifier Error",
    }

    # Group by step
    steps = sorted(set(q.step for q in quality_results))
    step_counts: dict[int, dict[str, int]] = {}
    for q in quality_results:
        sc = step_counts.setdefault(q.step, {})
        sc[q.status] = sc.get(q.status, 0) + 1

    all_statuses = ["usable", "invalid_reward", "invalid_empty", "invalid_format",
                    "invalid_truncated", "invalid_verifier"]

    margin_l, margin_r, margin_t, margin_b = 60, 30, 40, 80
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    n = len(steps)
    bar_w = min(60, plot_w / max(n, 1) - 10)
    max_total = max(sum(step_counts[s].values()) for s in steps) or 1

    # Axes
    chart.line(margin_l, margin_t, margin_l, margin_t + plot_h, stroke=theme.axis, sw=1)
    chart.line(margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h,
               stroke=theme.axis, sw=1)

    for frac in (0.25, 0.5, 0.75, 1.0):
        y = margin_t + plot_h * (1 - frac)
        chart.line(margin_l, y, margin_l + plot_w, y, stroke=theme.grid, sw=0.5, dash="4,4")
        chart.text(margin_l - 5, y + 4, f"{int(frac * max_total)}", size=9,
                   fill=theme.axis, anchor="end")

    # Stacked bars
    for i, step in enumerate(steps):
        x = margin_l + i * (plot_w / n) + (plot_w / n - bar_w) / 2
        y_bottom = margin_t + plot_h
        for status in all_statuses:
            count = step_counts[step].get(status, 0)
            if count == 0:
                continue
            h = count / max_total * plot_h
            color = status_colors.get(status, "#cccccc")
            chart.rect(x, y_bottom - h, bar_w, h, fill=color)
            y_bottom -= h

        chart.text(x + bar_w / 2, margin_t + plot_h + 15, f"Step {step}", size=10,
                   fill=theme.text, anchor="middle")

    chart.text(margin_l - 10, margin_t + plot_h / 2, "Count", size=11,
               fill=theme.text, anchor="end", rotate=-90)

    # Legend
    lx, ly = margin_l, height - 18
    for status in all_statuses:
        count_total = sum(step_counts[s].get(status, 0) for s in steps)
        if count_total == 0:
            continue
        color = status_colors.get(status, "#ccc")
        label = status_labels.get(status, status)
        chart.rect(lx, ly - 8, 12, 10, fill=color)
        chart.text(lx + 16, ly, label, size=9, fill=theme.text)
        lx += len(label) * 6 + 30

    return chart.render()


def chart_group_signal_heatmap(
    group_signals: list[GroupSignalResult],
    theme: Optional[ChartTheme] = None,
    width: int = 900,
    height: int = 400,
) -> str:
    """Chart 7: Group signal heatmap (step x group matrix).

    Each cell is colored by signal type, giving a bird's-eye view of
    the overall learning-signal landscape.
    """
    theme = theme or ChartTheme()
    chart = SVGChart(width, height, theme)
    chart.title("Group Signal Heatmap")

    if not group_signals:
        chart.text(width / 2, height / 2, "No group signal data available", size=14,
                   fill=theme.text, anchor="middle")
        return chart.render()

    signal_colors = {
        "effective_group": theme.effective,
        "constant_score": theme.constant,
        "near_constant_score": theme.near_constant,
        "insufficient_group": theme.insufficient,
        "group_unresolved": theme.unresolved,
    }
    signal_labels = {
        "effective_group": "Effective",
        "constant_score": "Constant",
        "near_constant_score": "Near-Const",
        "insufficient_group": "Insufficient",
        "group_unresolved": "Unresolved",
    }

    steps = sorted(set(g.step for g in group_signals))
    # Build matrix: step rows, group columns
    step_groups: dict[int, list[GroupSignalResult]] = {}
    for g in group_signals:
        step_groups.setdefault(g.step, []).append(g)

    max_groups_in_step = max(len(gs) for gs in step_groups.values())
    n_steps = len(steps)

    margin_l, margin_r, margin_t, margin_b = 60, 30, 40, 80
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    cell_w = plot_w / max(max_groups_in_step, 1)
    cell_h = plot_h / max(n_steps, 1)

    # Draw cells
    for row_i, step in enumerate(steps):
        groups_in_step = sorted(step_groups[step], key=lambda g: g.group_id)
        for col_i, g in enumerate(groups_in_step):
            x = margin_l + col_i * cell_w
            y = margin_t + row_i * cell_h
            color = signal_colors.get(g.signal_type, "#f0f0f0")
            chart.rect(x + 1, y + 1, cell_w - 2, cell_h - 2, fill=color, rx=2)

            # IPR value in cell if large enough
            if cell_w > 40 and cell_h > 25:
                chart.text(x + cell_w / 2, y + cell_h / 2 + 3,
                           _fmt(g.informative_pair_ratio, 2), size=8,
                           fill="white" if color != theme.insufficient else theme.text,
                           anchor="middle")

        # Step label (left)
        chart.text(margin_l - 5, margin_t + row_i * cell_h + cell_h / 2 + 3,
                   f"Step {step}", size=9, fill=theme.text, anchor="end")

    # Grid lines
    chart.line(margin_l, margin_t, margin_l, margin_t + plot_h, stroke=theme.axis, sw=1)
    chart.line(margin_l, margin_t + plot_h, margin_l + plot_w, margin_t + plot_h,
               stroke=theme.axis, sw=1)

    # Legend
    lx, ly = margin_l, height - 25
    for signal_type, color in signal_colors.items():
        label = signal_labels.get(signal_type, signal_type)
        chart.rect(lx, ly - 8, 12, 10, fill=color, rx=2)
        chart.text(lx + 16, ly, label, size=9, fill=theme.text)
        lx += len(label) * 6 + 30

    return chart.render()


def chart_invalid_prompt_cards(
    invalid_prompts: list[dict],
    theme: Optional[ChartTheme] = None,
    width: int = 900,
    height: int = 600,
) -> str:
    """Chart 8: Abnormal prompt cards.

    Visualizes exported invalid prompts as cards: truncated prompt text,
    diagnostic-type badge, mini reward bar, step label.
    """
    theme = theme or ChartTheme()
    chart = SVGChart(width, height, theme)
    chart.title("Abnormal Prompt Cards")

    if not invalid_prompts:
        chart.text(width / 2, height / 2, "No abnormal prompts exported", size=14,
                   fill=theme.text, anchor="middle")
        return chart.render()

    diag_colors = {
        "constant_score": theme.constant,
        "near_constant_score": theme.near_constant,
        "insufficient_group": theme.insufficient,
        "group_unresolved": theme.unresolved,
    }
    diag_labels = {
        "constant_score": "Constant Score",
        "near_constant_score": "Near-Constant",
        "insufficient_group": "Insufficient Group",
        "group_unresolved": "Unresolved Group",
    }

    margin_l, margin_r, margin_t = 20, 20, 40
    card_w = (width - margin_l - margin_r - 20) / 2
    card_h = 110
    gap_y = 15
    n_cols = 2
    n_rows = (len(invalid_prompts) + n_cols - 1) // n_cols

    # Adjust height
    needed_h = margin_t + n_rows * (card_h + gap_y) + 20
    if needed_h > height:
        height = needed_h
        chart = SVGChart(width, height, theme)
        chart.title("Abnormal Prompt Cards")

    for i, prompt_data in enumerate(invalid_prompts):
        col = i % n_cols
        row = i // n_cols
        x = margin_l + col * (card_w + 20)
        y = margin_t + row * (card_h + gap_y)

        diag_type = prompt_data.get("diagnostic_type", "unknown")
        color = diag_colors.get(diag_type, "#cccccc")
        label = diag_labels.get(diag_type, diag_type)

        # Card background
        chart.rect(x, y, card_w, card_h, fill="#fafafa", stroke=theme.grid, sw=1, rx=6)

        # Badge
        chart.rect(x + 8, y + 8, len(label) * 7 + 16, 18, fill=color, rx=9)
        chart.text(x + 16, y + 20, label, size=10, fill="white", weight="bold")

        # Step label
        steps = prompt_data.get("triggered_steps", [])
        step_text = f"Step {steps[0]}" if steps else "Step ?"
        chart.text(x + card_w - 10, y + 20, step_text, size=10,
                   fill=theme.text, anchor="end")

        # Prompt text (truncated)
        prompt_text = prompt_data.get("prompt", "")
        if len(prompt_text) > 55:
            prompt_text = prompt_text[:52] + "..."
        chart.text(x + 10, y + 42, f'"{_escape(prompt_text)}"', size=11,
                   fill=theme.text)

        # Diagnostic reason (truncated)
        reason = prompt_data.get("diagnostic_reason", "")
        if len(reason) > 70:
            reason = reason[:67] + "..."
        chart.text(x + 10, y + 58, _escape(reason), size=9, fill=theme.axis)

        # Mini reward bar
        rewards = prompt_data.get("rewards", [])
        bar_x = x + 10
        bar_y = y + 72
        bar_w = card_w - 20
        bar_h = 14
        chart.rect(bar_x, bar_y, bar_w, bar_h, fill="#e8e8e8", rx=3)

        if rewards:
            r_vals = [float(r) if r is not None else 0 for r in rewards]
            r_min, r_max = min(r_vals), max(r_vals)
            if r_max == r_min:
                r_max = r_min + 1
            r_range = r_max - r_min
            seg_w = bar_w / len(rewards)
            for j, r in enumerate(r_vals):
                val = (r - r_min) / r_range
                fill_color = color if val > 0.5 else "#bbb"
                chart.rect(bar_x + j * seg_w + 1, bar_y + 2,
                           seg_w - 2, bar_h - 4, fill=fill_color, rx=2)

        # Reward values text
        rewards_str = ", ".join(_fmt(float(r), 1) if r is not None else "null"
                                for r in rewards[:8])
        if len(rewards) > 8:
            rewards_str += "..."
        chart.text(bar_x, y + card_h - 5, f"Rewards: [{rewards_str}]", size=9,
                   fill=theme.axis)

    return chart.render()


# ── SVG export utility ──────────────────────────────────────────

def export_all_svgs(
    output_dir: str,
    step_aggs: list = None,
    quality_results: list = None,
    group_signals: list = None,
    diff_report=None,
    invalid_prompts: list = None,
    theme: Optional[ChartTheme] = None,
) -> list[str]:
    """Export all charts as standalone .svg files to *output_dir*.

    Returns list of written file paths.
    """
    import os
    step_aggs = step_aggs or []
    quality_results = quality_results or []
    group_signals = group_signals or []
    invalid_prompts = invalid_prompts or []
    written: list[str] = []

    chart_specs = [
        ("chart_step_signal.svg", chart_step_signal, step_aggs),
        ("chart_epoch_comparison.svg", chart_epoch_comparison, step_aggs),
        ("chart_rollout_quality.svg", chart_rollout_quality, quality_results),
        ("chart_group_reward_distribution.svg", chart_group_reward_distribution, group_signals),
        ("chart_rollout_quality_timeline.svg", chart_rollout_quality_timeline, quality_results),
        ("chart_group_signal_heatmap.svg", chart_group_signal_heatmap, group_signals),
    ]

    for filename, fn, data in chart_specs:
        if not data:
            continue
        svg_str = fn(data, theme=theme)
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg_str)
        written.append(path)

    # Difficulty chart (needs diff_report)
    if diff_report is not None:
        svg_str = chart_difficulty_stratification(diff_report, theme=theme)
        path = os.path.join(output_dir, "chart_difficulty_stratification.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg_str)
        written.append(path)

    # Invalid prompt cards
    if invalid_prompts:
        svg_str = chart_invalid_prompt_cards(invalid_prompts, theme=theme)
        path = os.path.join(output_dir, "chart_invalid_prompt_cards.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg_str)
        written.append(path)

    return written
