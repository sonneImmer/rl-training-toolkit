import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
METRIC = re.compile(r"([A-Za-z0-9_./-]+):([-+0-9.eE]+)")


def parse_metrics(log_path):
    rows = []
    for raw_line in log_path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw_line)
        match = re.search(r"step:(\d+) - ", line)
        if not match:
            continue
        values = {key: float(value) for key, value in METRIC.findall(line)}
        values["step"] = int(match.group(1))
        rows.append(values)
    return sorted({row["step"]: row for row in rows}.values(), key=lambda row: row["step"])


def load_rollouts(rollout_dir):
    by_step = defaultdict(list)
    for path in rollout_dir.glob("*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            by_step[int(row["step"])].append(row)
    return by_step


def load_effective_ratios(path):
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {
            int(row["step"]): float(row["effective_group_ratio"])
            for row in csv.DictReader(handle)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    metrics = parse_metrics(run / "train.log")
    rollouts = load_rollouts(run / "rollouts")
    effective = load_effective_ratios(run / "diagnostics" / "step_aggregations.csv")
    rows = []
    for metric in metrics:
        step = metric["step"]
        samples = rollouts.get(step, [])
        valid = sum(row.get("pred") not in (None, "[INVALID]") for row in samples)
        positive = sum(float(row.get("score", 0)) > 0 for row in samples)
        rows.append({
            "step": step,
            "rollouts": len(samples),
            "valid_answer_ratio": valid / len(samples) if samples else 0.0,
            "positive_reward_ratio": positive / len(samples) if samples else 0.0,
            "effective_group_ratio": effective.get(step),
            "reward_mean": metric.get("critic/score/mean"),
            "clip_ratio": metric.get("response_length/clip_ratio"),
            "response_length_mean": metric.get("response_length/mean"),
            "grad_norm": metric.get("actor/grad_norm"),
            "kl_loss": metric.get("actor/kl_loss"),
            "step_seconds": metric.get("timing_s/step"),
            "throughput": metric.get("perf/throughput"),
        })
    output = run / "report_materials"
    output.mkdir(exist_ok=True)
    fields = list(rows[0])
    with (output / "step_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {
        "run": str(run),
        "steps": len(rows),
        "rollouts": sum(row["rollouts"] for row in rows),
        "valid_answer_ratio": statistics.mean(row["valid_answer_ratio"] for row in rows),
        "positive_reward_ratio": statistics.mean(row["positive_reward_ratio"] for row in rows),
        "effective_group_ratio": statistics.mean(row["effective_group_ratio"] for row in rows),
        "reward_mean": statistics.mean(row["reward_mean"] for row in rows),
        "clip_ratio": statistics.mean(row["clip_ratio"] for row in rows),
        "grad_norm_min": min(row["grad_norm"] for row in rows),
        "grad_norm_max": max(row["grad_norm"] for row in rows),
        "kl_loss_first": rows[0]["kl_loss"],
        "kl_loss_last": rows[-1]["kl_loss"],
        "training_step_seconds": sum(row["step_seconds"] for row in rows),
        "note": "Each step samples different questions; reward changes across steps are not a controlled learning curve.",
    }
    (output / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
