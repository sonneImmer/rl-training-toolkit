#!/usr/bin/env python3
"""Build reproducible AMLSim experiment and PPT source material."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

STAGE_LABELS = {
    "base": "Base",
    "sft": "SFT (EOS fixed)",
    "grpo": "GRPO step 5",
    "ngrpo_step1": "NGRPO step 1",
    "ngrpo_step4": "NGRPO step 4",
    "ngrpo_step5": "NGRPO step 5",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(key: str, metrics: dict) -> dict:
    return {
        "stage": key,
        "label": STAGE_LABELS[key],
        "risk_accuracy": metrics["risk"]["accuracy"],
        "risk_precision": metrics["risk"]["precision"],
        "risk_recall": metrics["risk"]["recall"],
        "risk_f1": metrics["risk"]["f1"],
        "typology_accuracy": metrics["typology_accuracy"],
        "format_valid_rate": metrics["format_valid_rate"],
        "evidence_f1": metrics["evidence_f1"],
        "normal_accuracy": metrics["per_typology"]["NORMAL"]["accuracy"],
        "cycle_accuracy": metrics["per_typology"]["CYCLE"]["accuracy"],
        "fan_in_accuracy": metrics["per_typology"]["FAN_IN"]["accuracy"],
        "fan_out_accuracy": metrics["per_typology"]["FAN_OUT"]["accuracy"],
    }


def parse_stability(log_path: Path) -> list[dict]:
    pattern = re.compile(
        r"step:(?P<step>\d+).*?actor/entropy:(?P<entropy>[-+0-9.eE]+).*?"
        r"actor/kl_loss:(?P<kl>[-+0-9.eE]+).*?actor/pg_clipfrac:(?P<clip>[-+0-9.eE]+).*?"
        r"actor/grad_norm:(?P<grad>[-+0-9.eE]+).*?critic/score/mean:(?P<reward>[-+0-9.eE]+)"
    )
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            rows.append({"step": int(match["step"]), **{key: float(match[key]) for key in ("entropy", "kl", "clip", "grad", "reward")}})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def svg_line_chart(path: Path, series: dict[str, list[tuple[int, float]]], title: str, y_label: str) -> None:
    width, height = 960, 520
    left, top, right, bottom = 90, 60, 40, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [value for points in series.values() for _, value in points]
    low, high = min(0.0, min(values)), max(values)
    if high == low:
        high = low + 1
    colors = ["#1677ff", "#d4380d", "#389e0d", "#722ed1"]
    max_step = max(step for points in series.values() for step, _ in points)
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="22" fill="#111">{title}</text>']
    for tick in range(6):
        value = low + (high - low) * tick / 5
        y = top + plot_h - plot_h * tick / 5
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        elements.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="13" fill="#444">{value:.2f}</text>')
    for step in range(1, max_step + 1):
        x = left + plot_w * (step - 1) / max(1, max_step - 1)
        elements.append(f'<text x="{x:.1f}" y="{top+plot_h+28}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#444">{step}</text>')
    elements.append(f'<text x="24" y="{top+plot_h/2}" transform="rotate(-90 24 {top+plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#444">{y_label}</text>')
    for index, (name, points) in enumerate(series.items()):
        color = colors[index % len(colors)]
        coords = []
        for step, value in points:
            x = left + plot_w * (step - 1) / max(1, max_step - 1)
            y = top + plot_h * (high - value) / (high - low)
            coords.append((x, y))
        elements.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in coords)}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in coords:
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        lx, ly = left + 170 * index, height - 22
        elements.append(f'<line x1="{lx}" y1="{ly-5}" x2="{lx+28}" y2="{ly-5}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{lx+36}" y="{ly}" font-family="sans-serif" font-size="14" fill="#222">{name}</text>')
    elements.append('</svg>')
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def svg_bar_chart(path: Path, rows: list[dict]) -> None:
    width, height = 1100, 560
    left, top, right, bottom = 90, 60, 40, 150
    plot_w, plot_h = width-left-right, height-top-bottom
    keys = [("risk_accuracy", "Risk"), ("typology_accuracy", "Typology"), ("evidence_f1", "Evidence F1")]
    colors = ["#1677ff", "#389e0d", "#d4380d"]
    group_w = plot_w / len(rows)
    bar_w = group_w / 5
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="22">Independent test metrics (800 samples)</text>']
    for tick in range(6):
        value = tick / 5
        y = top + plot_h * (1-value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-family="sans-serif" font-size="13">{value:.1f}</text>')
    for index, row in enumerate(rows):
        center = left + group_w * (index + 0.5)
        for metric_index, (key, _) in enumerate(keys):
            value = row[key]
            x = center + (metric_index-1)*bar_w - bar_w/2
            y = top + plot_h*(1-value)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{plot_h*value:.1f}" fill="{colors[metric_index]}"/>')
        parts.append(f'<text x="{center:.1f}" y="{top+plot_h+25}" text-anchor="end" transform="rotate(-35 {center:.1f} {top+plot_h+25})" font-family="sans-serif" font-size="13">{row["label"]}</text>')
    for index, (_, label) in enumerate(keys):
        x = left + index*170
        parts.append(f'<rect x="{x}" y="{height-24}" width="18" height="12" fill="{colors[index]}"/>')
        parts.append(f'<text x="{x+26}" y="{height-13}" font-family="sans-serif" font-size="14">{label}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--grpo-run", type=Path, required=True)
    parser.add_argument("--ngrpo-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evaluation_dirs = {
        "base": "E0-base-full", "sft": "E1b-sft-eos-step30-full", "grpo": "E2-grpo-step5-full",
        "ngrpo_step1": "E3a-ngrpo-step1-full", "ngrpo_step4": "E3b-ngrpo-step4-full", "ngrpo_step5": "E3-ngrpo-step5-full",
    }
    metric_rows = [metric_row(key, load_json(args.evaluations / directory / "metrics.json")) for key, directory in evaluation_dirs.items()]
    write_csv(args.output / "metrics_comparison.csv", metric_rows)
    grpo_rollout = load_json(args.grpo_run / "analysis/rollout_summary.json")
    ngrpo_rollout = load_json(args.ngrpo_run / "analysis/rollout_summary.json")
    rollout_rows = []
    for algorithm, summaries in (("grpo", grpo_rollout), ("ngrpo", ngrpo_rollout)):
        for summary in summaries:
            rollout_rows.append({key: value for key, value in {"algorithm": algorithm, **summary}.items() if key not in {"expected_groups", "per_typology"}})
    write_csv(args.output / "rollout_trends.csv", rollout_rows)
    stability_rows = []
    for algorithm, run in (("grpo", args.grpo_run), ("ngrpo", args.ngrpo_run)):
        stability_rows.extend({"algorithm": algorithm, **row} for row in parse_stability(run / "train.log"))
    write_csv(args.output / "stability_metrics.csv", stability_rows)
    typology_rows = []
    for algorithm, run in (("grpo", args.grpo_run), ("ngrpo", args.ngrpo_run)):
        with (run / "analysis/rollout_typology_summary.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                typology_rows.append({"algorithm": algorithm, **row})
    write_csv(args.output / "rollout_typology_trends.csv", typology_rows)
    svg_bar_chart(args.output / "independent_metrics.svg", metric_rows)
    svg_line_chart(args.output / "rollout_exact_rate.svg", {
        "GRPO": [(row["step"], row["exact_rate"]) for row in grpo_rollout],
        "NGRPO": [(row["step"], row["exact_rate"]) for row in ngrpo_rollout],
    }, "Rollout exact-match trend", "Exact rate")
    svg_line_chart(args.output / "gradient_signal.svg", {
        "GRPO": [(row["step"], row["gradient_group_rate"]) for row in grpo_rollout],
        "NGRPO": [(row["step"], row["gradient_group_rate"]) for row in ngrpo_rollout],
    }, "Groups retaining relative learning signal", "Gradient group rate")

    by_stage = {row["stage"]: row for row in metric_rows}
    report = f"""# AMLSim反洗钱强化学习实验案例\n\n## 一、实验目标\n\n使用AMLSim公开类型定义构造可验证的反洗钱规则任务，验证“数据规则化 -> SFT格式对齐 -> GRPO诊断 -> NGRPO优化 -> 独立复评”的最小闭环。任务包含FAN_IN、FAN_OUT、CYCLE和NORMAL四类，训练集2400条、开发集400条、独立测试集800条。\n\n## 二、数据与奖励\n\n- 专家规则：FAN_IN为多源向中心账户汇入；FAN_OUT为中心账户向多目标分散转出；CYCLE为闭合转账环；NORMAL包含故意打断阈值的近似模式。\n- 输出：Risk、Typology、Evidence三行固定格式，不要求模型生成自由文本CoT。\n- 复合奖励范围[-4, 4]：风险判断占2分，类型判断占1分，证据集合F1占1分。重复Evidence ID按多重集合计数，避免重复ID钻奖励漏洞。\n- 限制：本实验使用AMLSim类型语义的受控生成器，并非运行原始AMLSim Java仿真全链路。\n\n## 三、关键发现\n\n1. Base模型风险准确率仅{by_stage['base']['risk_accuracy']:.2%}，主要问题是漏报，证明任务不是靠格式提示即可解决。\n2. 初始SFT曾出现“验证损失很低但生成失控”，定位到VERL SFT数据集把最终EOS从loss mask中移除。修复后，SFT风险准确率达到{by_stage['sft']['risk_accuracy']:.2%}，但NORMAL准确率只有{by_stage['sft']['normal_accuracy']:.2%}。\n3. 首次GRPO监控发现重复Evidence ID仍可得满分，说明Reward存在漏洞；修复后重跑，GRPO风险准确率达到{by_stage['grpo']['risk_accuracy']:.2%}，类型准确率{by_stage['grpo']['typology_accuracy']:.2%}。\n4. GRPO第5步Rollout准确率达到{grpo_rollout[-1]['exact_rate']:.2%}，但可产生相对梯度的group降为{grpo_rollout[-1]['gradient_group_rate']:.2%}，表明训练信号饱和。\n5. 依据该诊断切换NGRPO后，第1步可学习group占{ngrpo_rollout[0]['gradient_group_rate']:.2%}，高于GRPO首步的{grpo_rollout[0]['gradient_group_rate']:.2%}；最终风险准确率提高到{by_stage['ngrpo_step5']['risk_accuracy']:.2%}，误报由25条降到11条。\n6. NGRPO并非全面优于GRPO：最终类型准确率为{by_stage['ngrpo_step5']['typology_accuracy']:.2%}，低于GRPO的{by_stage['grpo']['typology_accuracy']:.2%}，其中FAN_OUT准确率从{by_stage['grpo']['fan_out_accuracy']:.2%}降到{by_stage['ngrpo_step5']['fan_out_accuracy']:.2%}。\n7. UI/训练内第4步FAN_OUT Rollout为100%全对，但独立测试仅{by_stage['ngrpo_step4']['fan_out_accuracy']:.2%}，说明小批次Rollout不能替代分类型独立评测。\n\n## 四、结论\n\n这次实验完成了可复现闭环，并验证了工具的实际价值：它不仅展示奖励上涨，还能发现EOS训练缺陷、Reward漏洞、全对/全错组导致的学习信号消失，以及总指标掩盖的业务类型退化。若业务目标优先降低漏报和误报，NGRPO step 5更好；若要求类型识别稳定，当前GRPO step 5更均衡。下一轮不应继续盲目加步数，而应提高FAN_IN/FAN_OUT反例覆盖或增加类型均衡约束后再训练。\n\n## 五、材料索引\n\n- `metrics_comparison.csv`：全部独立评测结果。\n- `rollout_trends.csv`：GRPO/NGRPO训练内Rollout趋势。\n- `rollout_typology_trends.csv`：按类型拆分的训练内趋势。\n- `stability_metrics.csv`：Entropy、KL、Grad Norm、Clip Fraction和Reward。\n- `independent_metrics.svg`、`rollout_exact_rate.svg`、`gradient_signal.svg`：PPT可直接使用的图表。\n"""
    (args.output / "EXPERIMENT_REPORT.md").write_text(report, encoding="utf-8")
    ppt = f"""# PPT汇报提纲：从Rollout诊断到算法选择\n\n## 第1页：业务问题\n- 目标：识别反洗钱风险、类型和证据交易。\n- 数据：AMLSim语义受控合成，4类，2400/400/800划分。\n- 判断标准：必须用独立测试集，不以训练Reward替代业务效果。\n\n## 第2页：Base与SFT\n- Base风险准确率：{by_stage['base']['risk_accuracy']:.2%}，严重漏报。\n- SFT修复EOS后：{by_stage['sft']['risk_accuracy']:.2%}，格式有效率{by_stage['sft']['format_valid_rate']:.2%}。\n- 剩余问题：NORMAL仅{by_stage['sft']['normal_accuracy']:.2%}，近似规则误报明显。\n\n## 第3页：UI发现两个问题\n- Reward漏洞：重复Evidence ID可异常得高分，修复后重跑。\n- GRPO信号饱和：第5步Rollout {grpo_rollout[-1]['exact_rate']:.2%}，但有效梯度group为0。\n- 图：`rollout_exact_rate.svg`、`gradient_signal.svg`。\n\n## 第4页：依据诊断切换NGRPO\n- Skill依据：当group大量全对/全错、相对优势归零时，NGRPO用最大奖励基线补充信号。\n- NGRPO首步有效梯度group：{ngrpo_rollout[0]['gradient_group_rate']:.2%}；GRPO首步：{grpo_rollout[0]['gradient_group_rate']:.2%}。\n- 训练过程未崩溃：第2步Grad Norm短时升高后快速回落，Response Clip始终为0。\n\n## 第5页：独立测试结果\n- GRPO：风险{by_stage['grpo']['risk_accuracy']:.2%}，类型{by_stage['grpo']['typology_accuracy']:.2%}，证据F1 {by_stage['grpo']['evidence_f1']:.2%}。\n- NGRPO：风险{by_stage['ngrpo_step5']['risk_accuracy']:.2%}，类型{by_stage['ngrpo_step5']['typology_accuracy']:.2%}，证据F1 {by_stage['ngrpo_step5']['evidence_f1']:.2%}。\n- 图：`independent_metrics.svg`。\n\n## 第6页：不能只讲“NGRPO更好”\n- NGRPO误报更少，但FAN_OUT从{by_stage['grpo']['fan_out_accuracy']:.2%}降到{by_stage['ngrpo_step5']['fan_out_accuracy']:.2%}。\n- 第4步训练内FAN_OUT为100%，独立测试仅{by_stage['ngrpo_step4']['fan_out_accuracy']:.2%}。\n- 结论：UI需要同时展示总体趋势、按类型趋势和独立评测。\n\n## 第7页：下一步优化\n- 增加FAN_IN/FAN_OUT方向性反例和边界样本。\n- 引入按类型最低准确率或宏平均约束，避免只优化风险二分类。\n- 每个保存点自动独立复评，并设置“总体上升但单类下降”的告警。\n"""
    (args.output / "PPT_MATERIAL.md").write_text(ppt, encoding="utf-8")
    provenance = {
        "evaluations": {key: str(args.evaluations / directory) for key, directory in evaluation_dirs.items()},
        "grpo_run": str(args.grpo_run), "ngrpo_run": str(args.ngrpo_run),
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
