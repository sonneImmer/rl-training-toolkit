#!/usr/bin/env python3
"""Build HBM comparison artifacts from RL-Insight Prometheus data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path


COLORS = {"deferred": "#c63c32", "per_micro_batch": "#167d67"}


def load_case(run_dir: Path) -> dict:
    timing = json.loads((run_dir / "case_timing.json").read_text(encoding="utf-8"))
    return {
        "run_dir": run_dir,
        "mode": timing["gradient_sync_mode"],
        "start": datetime.fromisoformat(timing["started_at"]).timestamp(),
        "end": datetime.fromisoformat(timing["finished_at"]).timestamp(),
        "exit_code": timing["exit_code"],
    }


def query_hbm(base_url: str, start: float, end: float) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "query": 'npu_chip_info_hbm_used_memory{job="npu-exporter"}',
            "start": f"{start:.3f}",
            "end": f"{end:.3f}",
            "step": "5",
        }
    )
    with urllib.request.urlopen(f"{base_url}/api/v1/query_range?{params}", timeout=30) as response:
        payload = json.load(response)
    if payload.get("status") != "success":
        raise RuntimeError(payload)
    return payload["data"]["result"]


def aggregate_hbm(series: list[dict], start: float) -> list[tuple[float, float, float]]:
    by_time: dict[float, list[float]] = {}
    for item in series:
        for timestamp, value in item["values"]:
            by_time.setdefault(float(timestamp), []).append(float(value) / 1024.0)
    return [
        (timestamp - start, max(values), sum(values) / len(values))
        for timestamp, values in sorted(by_time.items())
    ]


def parse_train_metrics(log_path: Path) -> list[dict[str, float]]:
    keys = (
        "timing_s/update_actor",
        "timing_s/step",
        "perf/max_memory_allocated_gb",
        "perf/max_memory_reserved_gb",
        "perf/throughput",
    )
    rows = []
    if not log_path.is_file():
        return rows
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "step:" not in line:
            continue
        row: dict[str, float] = {}
        step_match = re.search(r"step:(\d+)", line)
        if step_match:
            row["step"] = float(step_match.group(1))
        for key in keys:
            match = re.search(rf"{re.escape(key)}:([-+0-9.eE]+)", line)
            if match:
                row[key] = float(match.group(1))
        if len(row) > 1:
            rows.append(row)
    return rows


def mean_metric(rows: list[dict[str, float]], key: str) -> float | None:
    values = [row[key] for row in rows if key in row and math.isfinite(row[key])]
    return sum(values) / len(values) if values else None


def write_svg(cases: list[dict], output: Path) -> None:
    width, height = 1200, 620
    left, right, top, bottom = 88, 36, 54, 78
    plot_w, plot_h = width - left - right, height - top - bottom
    max_x = max((row[0] for case in cases for row in case["hbm"]), default=1.0)
    max_y = max((row[1] for case in cases for row in case["hbm"]), default=1.0)
    max_y = max(8.0, math.ceil(max_y / 8.0) * 8.0)

    def x(value: float) -> float:
        return left + value / max_x * plot_w

    def y(value: float) -> float:
        return top + plot_h - value / max_y * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="88" y="31" font-family="sans-serif" font-size="22" font-weight="700">Actor Update HBM: gradient synchronization comparison</text>',
    ]
    for tick in range(6):
        value = max_y * tick / 5
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="13">{value:.0f}</text>')
    for tick in range(7):
        value = max_x * tick / 6
        xx = x(value)
        parts.append(f'<text x="{xx:.1f}" y="{height-bottom+28}" text-anchor="middle" font-family="sans-serif" font-size="13">{value:.0f}</text>')
    parts.append(f'<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20 {top+plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="14">Peak HBM across 8 NPUs (GiB)</text>')
    parts.append(f'<text x="{left+plot_w/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">Seconds since experiment start</text>')
    for index, case in enumerate(cases):
        color = COLORS[case["mode"]]
        points = " ".join(f"{x(row[0]):.1f},{y(row[1]):.1f}" for row in case["hbm"])
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        legend_y = 28 + index * 24
        parts.append(f'<line x1="835" y1="{legend_y}" x2="863" y2="{legend_y}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text x="871" y="{legend_y+5}" font-family="sans-serif" font-size="14">{case["mode"]}</text>')
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_bar_svg(cases: list[dict], output: Path) -> None:
    width, height = 980, 580
    left, top, plot_w, plot_h = 90, 70, 830, 400
    metrics = (
        ("RL-Insight HBM", "peak_hbm_gib"),
        ("VERL Actor allocated", "peak_actor_allocated_gib"),
    )
    max_value = max(case[key] for _, key in metrics for case in cases if case[key] is not None)
    max_value = math.ceil(max_value / 10.0) * 10.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="90" y="34" font-family="sans-serif" font-size="22" font-weight="700">Peak memory comparison</text>',
    ]
    for tick in range(8):
        value = max_value * tick / 7
        y = top + plot_h - value / max_value * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="13">{value:.0f}</text>')
    group_width, bar_width = plot_w / len(metrics), 96
    for metric_index, (label, key) in enumerate(metrics):
        center = left + group_width * (metric_index + 0.5)
        for case_index, case in enumerate(cases):
            value = case[key]
            x = center + (case_index - 0.5) * (bar_width + 18) - bar_width / 2
            bar_height = value / max_value * plot_h
            y = top + plot_h - bar_height
            color = COLORS[case["mode"]]
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}" rx="4"/>')
            parts.append(f'<text x="{x+bar_width/2:.1f}" y="{y-9:.1f}" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="700">{value:.2f}</text>')
        parts.append(f'<text x="{center:.1f}" y="{top+plot_h+34}" text-anchor="middle" font-family="sans-serif" font-size="15">{label}</text>')
    for index, case in enumerate(cases):
        x = 570 + index * 180
        color = COLORS[case["mode"]]
        parts.append(f'<rect x="{x}" y="24" width="18" height="18" fill="{color}" rx="3"/>')
        parts.append(f'<text x="{x+27}" y="38" font-family="sans-serif" font-size="14">{case["mode"]}</text>')
    parts.append('<text x="22" y="270" transform="rotate(-90 22 270)" text-anchor="middle" font-family="sans-serif" font-size="14">Peak memory (GiB)</text>')
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def fmt(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def write_html(cases: list[dict], output: Path) -> None:
    baseline, optimized = cases
    actor_reduction = baseline["peak_actor_allocated_gib"] - optimized["peak_actor_allocated_gib"]
    actor_reduction_pct = actor_reduction / baseline["peak_actor_allocated_gib"] * 100
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RL-Insight Actor Update 显存案例</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #17212b; background: #f4f6f8; }}
    main {{ width: min(1180px, calc(100% - 40px)); margin: 32px auto 56px; }}
    h1 {{ font-size: 30px; margin: 0 0 8px; }}
    .note {{ color: #53606d; margin-bottom: 24px; }}
    .summary {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px; }}
    .metric {{ background: #fff; border: 1px solid #dce2e8; border-radius: 6px; padding: 18px; }}
    .metric strong {{ display: block; font-size: 26px; margin-top: 8px; }}
    section {{ background: #fff; border: 1px solid #dce2e8; border-radius: 6px; padding: 20px; margin-top: 16px; }}
    img {{ display: block; width: 100%; height: auto; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #e4e8ec; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    @media (max-width: 760px) {{ .summary {{ grid-template-columns: 1fr; }} main {{ width: min(100% - 20px, 1180px); }} }}
  </style>
</head>
<body><main>
  <h1>RL-Insight Actor Update 显存案例</h1>
  <p class="note">8 × Ascend 910B3 · Qwen3 8.19B · GRPO · 同参数仅切换梯度同步方式</p>
  <div class="summary">
    <div class="metric">Actor 峰值下降<strong>{actor_reduction:.2f} GiB</strong></div>
    <div class="metric">相对降幅<strong>{actor_reduction_pct:.1f}%</strong></div>
    <div class="metric">RL-Insight 采样峰值差<strong>{baseline['peak_hbm_gib'] - optimized['peak_hbm_gib']:.2f} GiB</strong></div>
  </div>
  <section><table>
    <thead><tr><th>模式</th><th>RL-Insight HBM</th><th>Actor 分配峰值</th><th>Actor Update</th><th>Step</th><th>吞吐</th></tr></thead>
    <tbody>
      <tr><td>deferred（故障注入）</td><td>{baseline['peak_hbm_gib']:.2f} GiB</td><td>{baseline['peak_actor_allocated_gib']:.2f} GiB</td><td>{baseline['mean_actor_update_s']:.2f} s</td><td>{baseline['mean_step_s']:.2f} s</td><td>{baseline['mean_throughput']:.2f}</td></tr>
      <tr><td>per_micro_batch（原生优化）</td><td>{optimized['peak_hbm_gib']:.2f} GiB</td><td>{optimized['peak_actor_allocated_gib']:.2f} GiB</td><td>{optimized['mean_actor_update_s']:.2f} s</td><td>{optimized['mean_step_s']:.2f} s</td><td>{optimized['mean_throughput']:.2f}</td></tr>
    </tbody>
  </table></section>
  <section><img src="peak_memory_comparison.svg" alt="Peak memory comparison"></section>
  <section><img src="hbm_timeline.svg" alt="HBM timeline"></section>
  <section><strong>解释边界</strong><p>deferred 是为复现问题而注入的延迟同步故障；当前固定镜像默认采用 per_micro_batch。NPU Exporter 每 10 秒采样，整机曲线可能错过短时 backward 尖峰，因此以 VERL Actor 分配峰值确认瞬时差异。</p></section>
</main></body></html>
"""
    output.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prometheus", default="http://127.0.0.1:19090")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cases = [load_case(args.baseline.resolve()), load_case(args.optimized.resolve())]
    for case in cases:
        case["hbm"] = aggregate_hbm(query_hbm(args.prometheus, case["start"], case["end"]), case["start"])
        metrics = parse_train_metrics(case["run_dir"] / "train.log")
        case["peak_hbm_gib"] = max((row[1] for row in case["hbm"]), default=None)
        case["peak_actor_allocated_gib"] = max(
            (row["perf/max_memory_allocated_gb"] for row in metrics if "perf/max_memory_allocated_gb" in row),
            default=None,
        )
        case["mean_actor_update_s"] = mean_metric(metrics, "timing_s/update_actor")
        case["mean_step_s"] = mean_metric(metrics, "timing_s/step")
        case["mean_throughput"] = mean_metric(metrics, "perf/throughput")

    fields = ("mode", "exit_code", "peak_hbm_gib", "peak_actor_allocated_gib", "mean_actor_update_s", "mean_step_s", "mean_throughput", "run_dir")
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({key: case.get(key) for key in fields})
    write_svg(cases, args.output / "hbm_timeline.svg")
    write_bar_svg(cases, args.output / "peak_memory_comparison.svg")
    write_html(cases, args.output / "index.html")

    baseline, optimized = cases
    reduction = None
    if baseline["peak_hbm_gib"] and optimized["peak_hbm_gib"]:
        reduction = baseline["peak_hbm_gib"] - optimized["peak_hbm_gib"]
    actor_reduction = baseline["peak_actor_allocated_gib"] - optimized["peak_actor_allocated_gib"]
    actor_reduction_pct = actor_reduction / baseline["peak_actor_allocated_gib"] * 100
    report = f"""# RL-Insight Actor Update 显存案例

## 实验定义

- 基线 `deferred`：人为注入延迟梯度同步，仅用于复现官方案例的故障机制，不代表当前镜像的原生缺陷。
- 优化 `per_micro_batch`：当前固定镜像的原生行为，每个 micro-batch 正常同步梯度。
- 监控：RL-Insight 0.3.0 管理 Prometheus，采集 8 张 Ascend 910B3 的 NPU Exporter HBM 指标。
- 对照：同一模型、数据、随机种子和 GRPO 参数，仅改变梯度同步模式。

## 结果

| 模式 | 退出码 | RL-Insight 单卡峰值 HBM | VERL Actor 分配峰值 | Actor Update 平均耗时 | Step 平均耗时 | 平均吞吐 |
|---|---:|---:|---:|---:|---:|---:|
| deferred | {baseline['exit_code']} | {fmt(baseline['peak_hbm_gib'], ' GiB')} | {fmt(baseline['peak_actor_allocated_gib'], ' GiB')} | {fmt(baseline['mean_actor_update_s'], ' s')} | {fmt(baseline['mean_step_s'], ' s')} | {fmt(baseline['mean_throughput'])} |
| per_micro_batch | {optimized['exit_code']} | {fmt(optimized['peak_hbm_gib'], ' GiB')} | {fmt(optimized['peak_actor_allocated_gib'], ' GiB')} | {fmt(optimized['mean_actor_update_s'], ' s')} | {fmt(optimized['mean_step_s'], ' s')} | {fmt(optimized['mean_throughput'])} |

RL-Insight HBM 峰值差值：**{fmt(reduction, ' GiB')}**。VERL Actor 分配峰值下降：**{actor_reduction:.2f} GiB（{actor_reduction_pct:.1f}%）**。

RL-Insight 的 NPU Exporter 当前按 10 秒采样，可能错过短时 backward 尖峰；因此整机 HBM 曲线用于观察趋势，VERL 的 `perf/max_memory_allocated_gb` 用于确认 Actor Update 瞬时峰值。

![HBM timeline](hbm_timeline.svg)

![Peak memory comparison](peak_memory_comparison.svg)

## 结论边界

本案例验证“延迟同步保留未分片梯度，会抬高 Actor backward 显存峰值”的机制。当前 VERL 0.7.0 镜像默认没有该开关，因此优化组是现状，基线组是可回退的受控故障注入。模型规模、FSDP 版本和 NPU 数量不同，不能期待复刻官方 64GB 到 36GB 的绝对数值，只比较同环境下的相对变化。
"""
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    print(args.output / "REPORT.md")


if __name__ == "__main__":
    main()
