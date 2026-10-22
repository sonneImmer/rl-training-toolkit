#!/usr/bin/env python3
"""Deterministically evaluate AML risk, typology, and evidence quality."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .amlsim_reward import evidence_metrics, parse_output
except ImportError:
    from amlsim_reward import evidence_metrics, parse_output


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_metrics(records: list[dict]) -> dict:
    risk_tp = risk_fp = risk_tn = risk_fn = risk_invalid = 0
    typology_correct = format_valid = 0
    evidence_precision = evidence_recall = evidence_f1 = 0.0
    confusion = defaultdict(Counter)
    per_typology = defaultdict(lambda: {"samples": 0, "correct": 0})
    for record in records:
        expected = record["expected"]
        predicted = record["parsed"]
        expected_positive = expected["risk"] == "SUSPICIOUS"
        predicted_positive = predicted["risk"] == "SUSPICIOUS"
        if predicted["risk"] not in {"SUSPICIOUS", "NORMAL"}:
            risk_invalid += 1
            if expected_positive:
                risk_fn += 1
            else:
                risk_fp += 1
        elif expected_positive and predicted_positive:
            risk_tp += 1
        elif not expected_positive and predicted_positive:
            risk_fp += 1
        elif not expected_positive and not predicted_positive and predicted["risk"] == "NORMAL":
            risk_tn += 1
        else:
            risk_fn += 1
        typology_correct += predicted["typology"] == expected["typology"]
        format_valid += predicted["format_valid"]
        precision, recall, f1 = evidence_metrics(predicted["evidence"], expected["evidence"])
        evidence_precision += precision
        evidence_recall += recall
        evidence_f1 += f1
        confusion[expected["typology"]][predicted["typology"] or "INVALID"] += 1
        item = per_typology[expected["typology"]]
        item["samples"] += 1
        item["correct"] += predicted["typology"] == expected["typology"]
    total = len(records)
    risk_precision = safe_div(risk_tp, risk_tp + risk_fp)
    risk_recall = safe_div(risk_tp, risk_tp + risk_fn)
    risk_f1 = safe_div(2 * risk_precision * risk_recall, risk_precision + risk_recall)
    risk_f2 = safe_div(5 * risk_precision * risk_recall, 4 * risk_precision + risk_recall)
    return {
        "samples": total,
        "risk": {
            "true_positive": risk_tp, "false_positive": risk_fp, "true_negative": risk_tn, "false_negative": risk_fn, "invalid_output": risk_invalid,
            "accuracy": safe_div(risk_tp + risk_tn, total), "precision": risk_precision, "recall": risk_recall,
            "f1": risk_f1, "f2": risk_f2,
        },
        "typology_accuracy": safe_div(typology_correct, total),
        "format_valid_rate": safe_div(format_valid, total),
        "evidence_precision": safe_div(evidence_precision, total),
        "evidence_recall": safe_div(evidence_recall, total),
        "evidence_f1": safe_div(evidence_f1, total),
        "typology_confusion": {key: dict(value) for key, value in confusion.items()},
        "per_typology": {key: {**value, "accuracy": safe_div(value["correct"], value["samples"])} for key, value in per_typology.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    import pandas as pd
    from vllm import LLM, SamplingParams

    frame = pd.read_parquet(args.data)
    if args.limit:
        frame = frame.head(args.limit)
    engine = LLM(model=str(args.model), tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=False, enforce_eager=True, max_model_len=args.max_model_len, gpu_memory_utilization=0.75)
    tokenizer = engine.get_tokenizer()
    prompts, expected_rows = [], []
    for _, row in frame.iterrows():
        conversation = row["prompt"].tolist() if hasattr(row["prompt"], "tolist") else list(row["prompt"])
        prompts.append(tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True))
        expected_rows.append(json.loads(row["reward_model"]["ground_truth"]))
    outputs = engine.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens))
    responses = [output.outputs[0].text for output in outputs]
    records = []
    for (_, row), expected, response in zip(frame.iterrows(), expected_rows, responses):
        parsed = parse_output(response)
        records.append({"sample_id": row["extra_info"]["sample_id"], "expected": expected, "parsed": parsed, "response": response})
    metrics = calculate_metrics(records)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
