"""Dependency-free HTML rendering for the Jupyter experiment monitor."""
from __future__ import annotations

import html
from typing import Optional

from .experiment_store import ExperimentStore


RULE_NAMES = {
    "explore_collapse": "探索能力收缩",
    "kl_drift": "KL 漂移",
    "clip_saturation": "裁剪饱和",
    "gradient_spike": "梯度尖峰",
    "output_truncation": "输出截断",
    "reward_signal_degradation": "奖励信号退化",
    "performance_degradation": "训练性能下降",
}

STATUS_NAMES = {
    "running": "运行中",
    "completed": "已完成",
    "completed_with_errors": "完成但有尾部错误",
    "failed": "失败",
    "stopped": "停止更新",
    "unreadable": "不可读取",
}


def _escape(value: object) -> str:
    return html.escape("-" if value is None else str(value))


def _number(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _percent(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return str(value)


def _line_chart(rows: list[dict], series: list[tuple[str, str, str]], title: str) -> str:
    width, height = 760, 210
    left, top, right, bottom = 46, 28, 16, 30
    values = [
        float(row[key])
        for key, _, _ in series
        for row in rows
        if row.get(key) is not None
    ]
    if not rows or not values:
        return f'<div class="chart"><strong>{_escape(title)}</strong><p>暂无数据</p></div>'
    low, high = min(values), max(values)
    if low == high:
        low -= 0.5
        high += 0.5
    min_step = min(int(row["step"]) for row in rows)
    max_step = max(int(row["step"]) for row in rows)
    step_span = max(max_step - min_step, 1)
    inner_width = width - left - right
    inner_height = height - top - bottom
    polylines = []
    legends = []
    for index, (key, label, color) in enumerate(series):
        points = []
        for row in rows:
            if row.get(key) is None:
                continue
            x = left + (int(row["step"]) - min_step) / step_span * inner_width
            y = top + (high - float(row[key])) / (high - low) * inner_height
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            polylines.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.3" points="{" ".join(points)}"/>'
            )
        legend_x = left + index * 150
        legends.append(
            f'<line x1="{legend_x}" y1="12" x2="{legend_x + 16}" y2="12" stroke="{color}" stroke-width="3"/>'
            f'<text x="{legend_x + 21}" y="16" font-size="11">{_escape(label)}</text>'
        )
    return (
        f'<div class="chart"><strong>{_escape(title)}</strong>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">'
        f'<line x1="{left}" y1="{top + inner_height}" x2="{width - right}" y2="{top + inner_height}" stroke="#bcc5d0"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + inner_height}" stroke="#bcc5d0"/>'
        f'{"".join(polylines)}{"".join(legends)}'
        f'<text x="3" y="{top + 4}" font-size="10">{high:.3f}</text>'
        f'<text x="3" y="{top + inner_height}" font-size="10">{low:.3f}</text>'
        f'<text x="{left}" y="{height - 7}" font-size="10">step {min_step}</text>'
        f'<text x="{width - right - 58}" y="{height - 7}" font-size="10">step {max_step}</text>'
        "</svg></div>"
    )


def _metric_card(label: str, value: object) -> str:
    return f'<div class="metric"><span>{_escape(label)}</span><strong>{_escape(value)}</strong></div>'


def _rule_html(rule: dict) -> str:
    status_names = {"triggered": "已触发", "not_triggered": "未触发", "not_evaluated": "未评估"}
    details = []
    for label, key in (
        ("原因", "reason"), ("说明", "description"), ("指标", "metric_name"),
        ("当前窗口", "trigger_value"), ("基线窗口", "baseline_value"), ("阈值", "threshold"),
        ("窗口大小", "window_size"), ("基线长度", "baseline_window"),
    ):
        if rule.get(key) is not None:
            details.append(f"<div><b>{_escape(label)}：</b>{_escape(rule[key])}</div>")
    if rule.get("window_steps"):
        details.append(f'<div><b>证据 step：</b>{_escape(", ".join(map(str, rule["window_steps"])))}</div>')
    hints = rule.get("investigation_hints", [])
    if hints:
        details.append(f'<div><b>建议检查：</b>{_escape("；".join(hints))}</div>')
    name = RULE_NAMES.get(rule.get("rule_name"), rule.get("rule_name", "未知规则"))
    summary = rule.get("reason") or rule.get("description") or "点击展开详情"
    return (
        f'<details class="rule {rule.get("status", "not_evaluated")}">'
        f'<summary><b>{_escape(name)}</b><span>{_escape(status_names.get(rule.get("status"), rule.get("status")))}</span>'
        f'<small>{_escape(summary)}</small></summary><div class="rule-detail">{"".join(details) or "暂无更多信息"}</div></details>'
    )


def _comparison_html(store: ExperimentStore, selected_id: str, compare_id: Optional[str]) -> str:
    if not compare_id or compare_id == selected_id:
        return '<p class="muted">设置 COMPARE_EXPERIMENT_ID 后显示参数和指标差异。</p>'
    comparison = store.compare_experiments([selected_id, compare_id])
    if not comparison:
        return '<p class="muted">对比实验不存在。</p>'
    names = [item.get("name") or item["experiment_id"] for item in comparison["experiments"]]
    changed = [row for row in comparison["config_rows"] if row["changed"]]
    config_rows = "".join(
        f'<tr><td>{_escape(row["key"])}</td>{"".join(f"<td>{_escape(value)}</td>" for value in row["values"])}</tr>'
        for row in changed
    ) or '<tr><td colspan="3">配置没有差异，或日志未记录完整启动参数。</td></tr>'
    metric_rows = "".join(
        f'<tr><td>{_escape(row["key"])}</td>{"".join(f"<td>{_number(value)}</td>" for value in row["values"])}<td>{_number(row["delta"])}</td></tr>'
        for row in comparison["latest_metric_rows"]
    ) or '<tr><td colspan="4">暂无可比较指标。</td></tr>'
    combined = {}
    for index, experiment in enumerate(comparison["series"]):
        for row in experiment["steps"]:
            combined.setdefault(row["step"], {"step": row["step"]})[f"reward_{index}"] = row.get("reward_mean")
            combined[row["step"]][f"signal_{index}"] = row.get("effective_group_ratio")
    aligned_rows = [combined[step] for step in sorted(combined)]
    return (
        '<h4>参数差异</h4><div class="table-wrap"><table><thead><tr><th>参数</th>'
        f'{"".join(f"<th>{_escape(name)}</th>" for name in names)}</tr></thead><tbody>{config_rows}</tbody></table></div>'
        '<h4>最新指标</h4><div class="table-wrap"><table><thead><tr><th>指标</th>'
        f'{"".join(f"<th>{_escape(name)}</th>" for name in names)}<th>差值</th></tr></thead><tbody>{metric_rows}</tbody></table></div>'
        + _line_chart(aligned_rows, [("reward_0", names[0], "#176b5b"), ("reward_1", names[1], "#a22b2b")], "Reward 对齐趋势")
        + _line_chart(aligned_rows, [("signal_0", names[0], "#176b5b"), ("signal_1", names[1], "#a22b2b")], "有效 group 对齐趋势")
    )


def render_experiment_console(
    store: ExperimentStore,
    selected_id: Optional[str] = None,
    compare_id: Optional[str] = None,
    refresh: bool = True,
    rollout_limit: int = 6,
) -> str:
    experiments = store.refresh_all() if refresh else store.list_experiments()
    if not experiments:
        return '<div class="rl-console"><h2>没有发现实验</h2><p>RUNS_ROOT 下需要存在包含 train.log 的运行目录。</p></div>'
    ids = {item["experiment_id"] for item in experiments}
    if selected_id not in ids:
        selected_id = experiments[0]["experiment_id"]
    item = store.get_experiment(selected_id)
    if not item:
        return '<div class="rl-console"><h2>实验快照不可读取</h2></div>'
    rows = item.get("steps", [])
    latest = item.get("latest_metrics", {})
    diagnostics = item.get("diagnostics", {})
    evaluations = diagnostics.get("stability", {}).get("evaluations", [])
    progress = item.get("progress", {})
    quality = latest.get("rollout_quality_counts", {})

    experiment_rows = "".join(
        f'<tr class="{"selected" if exp["experiment_id"] == selected_id else ""}"><td>{_escape(exp.get("name"))}</td>'
        f'<td>{_escape(STATUS_NAMES.get(exp.get("status"), exp.get("status")))}</td><td>{_escape(exp.get("config", {}).get("algorithm"))}</td>'
        f'<td>{_escape(exp.get("progress", {}).get("current_step"))}</td><td><code>{_escape(exp["experiment_id"])}</code></td></tr>'
        for exp in experiments
    )
    cards = "".join([
        _metric_card("训练进度", f'{progress.get("current_step", "-")}/{progress.get("total_steps", "-")}'),
        _metric_card("平均 Reward", _number(latest.get("reward_mean"))),
        _metric_card("可用 rollout", _percent(latest.get("rollout_usable_ratio"))),
        _metric_card("有效 group", _percent(latest.get("effective_group_ratio"))),
        _metric_card("低信息 group", _percent(latest.get("low_information_group_ratio"))),
        _metric_card("可区分 pair", _percent(latest.get("mean_informative_pair_ratio"))),
        _metric_card("KL", _number(latest.get("kl_loss"))),
        _metric_card("Gradient Norm", _number(latest.get("grad_norm"))),
    ])
    quality_cards = "".join(_metric_card(key, value) for key, value in sorted(quality.items()))
    rules = "".join(_rule_html(rule) for rule in evaluations) or '<p class="muted">暂无稳定性规则结果。</p>'
    step_rows = "".join(
        f'<tr><td>{row.get("step")}</td><td>{row.get("rollouts", 0)}</td><td>{_percent(row.get("rollout_usable_ratio"))}</td>'
        f'<td>{_percent(row.get("effective_group_ratio"))}</td><td>{_percent(row.get("low_information_group_ratio"))}</td>'
        f'<td>{_percent(row.get("mean_informative_pair_ratio"))}</td><td>{_number(row.get("reward_mean"))}</td>'
        f'<td>{_number(row.get("kl_loss"))}</td><td>{_number(row.get("grad_norm"))}</td></tr>'
        for row in rows[-30:]
    )
    samples = store.get_rollouts(selected_id, progress.get("current_step"), rollout_limit)
    sample_html = "".join(
        f'<details class="sample"><summary>#{index} reward={_escape(record.get("score", record.get("reward")))} '
        f'acc={_escape(record.get("acc"))} pred={_escape(record.get("pred"))}</summary>'
        f'<pre>PROMPT\n{_escape(record.get("input", record.get("prompt", "")))}\n\nOUTPUT\n{_escape(record.get("output", record.get("response", "")))}</pre></details>'
        for index, record in enumerate(samples, 1)
    ) or '<p class="muted">最新 step 没有可读取的 rollout。</p>'
    config = item.get("config", {})
    config_html = "".join(_metric_card(key, value) for key, value in config.items())

    return f'''<style>
.rl-console{{font-family:Inter,"Noto Sans SC","Microsoft YaHei",system-ui;color:#18202a;background:#eef1f4;padding:16px;line-height:1.45}}
.rl-console h2{{margin:0;font-size:22px}}.rl-console h3{{font-size:14px;margin:0}}.rl-console h4{{font-size:12px;color:#657180;margin:14px 0 7px}}
.meta,.muted{{color:#657180;font-size:12px}}.band{{margin-top:14px;border:1px solid #bcc5d0;background:#fff}}.band-head{{padding:10px 12px;border-bottom:1px solid #d9dee5;background:#f7f8fa}}
.content{{padding:12px}}.metrics{{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:1px;background:#d9dee5}}.metric{{padding:10px 12px;background:#fff;min-width:0}}
.metric span{{display:block;color:#657180;font-size:11px}}.metric strong{{display:block;margin-top:3px;font-size:17px;overflow-wrap:anywhere}}.charts{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
.chart{{min-width:0}}.chart svg{{display:block;width:100%;height:210px;border:1px solid #d9dee5;margin-top:6px}}.rules{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}}
.rule{{border-left:3px solid #9a5a08;background:#fff3df}}.rule.triggered{{border-color:#a22b2b;background:#fdeaea}}.rule.not_triggered{{border-color:#176b5b;background:#e5f2ef}}
.rule summary{{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:3px 10px;padding:9px;cursor:pointer}}.rule summary small{{grid-column:1/-1;color:#657180}}.rule-detail{{padding:0 12px 10px 25px;color:#657180;font-size:11px}}
.table-wrap{{overflow:auto;max-height:330px;border:1px solid #d9dee5}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:7px 9px;border-bottom:1px solid #d9dee5;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}}tr.selected{{background:#e5f2ef}}code{{font-size:11px}}.sample{{border-bottom:1px solid #d9dee5;padding:8px}}pre{{white-space:pre-wrap;max-height:320px;overflow:auto;background:#f7f8fa;padding:10px}}
@media(max-width:900px){{.metrics{{grid-template-columns:repeat(2,1fr)}}.charts,.rules{{grid-template-columns:1fr}}}}
</style><div class="rl-console">
<h2>{_escape(item.get("name") or item["experiment_id"])}</h2><div class="meta">{_escape(item["experiment_id"])} · {_escape(STATUS_NAMES.get(item.get("status"), item.get("status")))}</div>
<section class="band"><div class="band-head"><h3>实验目录</h3></div><div class="content"><div class="table-wrap"><table><thead><tr><th>实验</th><th>状态</th><th>算法</th><th>step</th><th>实验 ID</th></tr></thead><tbody>{experiment_rows}</tbody></table></div><p class="muted">在配置单元格中设置 SELECTED_EXPERIMENT_ID 和 COMPARE_EXPERIMENT_ID。</p></div></section>
<section class="band"><div class="band-head"><h3>训练概览</h3></div><div class="metrics">{cards}</div></section>
<section class="band"><div class="band-head"><h3>Rollout 质量</h3></div><div class="metrics">{quality_cards or _metric_card("状态", "暂无数据")}</div></section>
<section class="band"><div class="band-head"><h3>趋势图</h3></div><div class="content charts">
{_line_chart(rows, [("reward_mean", "reward", "#176b5b"), ("rollout_usable_ratio", "可用 rollout", "#285a8e")], "Reward / Rollout 质量")}
{_line_chart(rows, [("effective_group_ratio", "有效 group", "#176b5b"), ("low_information_group_ratio", "低信息 group", "#a22b2b"), ("mean_informative_pair_ratio", "可区分 pair", "#285a8e")], "组内学习信号")}
{_line_chart(rows, [("kl_loss", "KL", "#8c4f15"), ("entropy", "entropy", "#5d4a8d")], "策略趋势")}
{_line_chart(rows, [("grad_norm", "grad norm", "#a22b2b")], "梯度趋势")}</div></section>
<section class="band"><div class="band-head"><h3>稳定性规则</h3></div><div class="content rules">{rules}</div></section>
<section class="band"><div class="band-head"><h3>最近步骤</h3></div><div class="table-wrap"><table><thead><tr><th>step</th><th>rollout</th><th>可用</th><th>有效 group</th><th>低信息</th><th>pair</th><th>reward</th><th>KL</th><th>grad</th></tr></thead><tbody>{step_rows}</tbody></table></div></section>
<section class="band"><div class="band-head"><h3>实验参数</h3></div><div class="metrics">{config_html}</div></section>
<section class="band"><div class="band-head"><h3>实验参数与指标对比</h3></div><div class="content">{_comparison_html(store, selected_id, compare_id)}</div></section>
<section class="band"><div class="band-head"><h3>最新 Rollout 证据</h3></div><div class="content">{sample_html}</div></section>
</div>'''
