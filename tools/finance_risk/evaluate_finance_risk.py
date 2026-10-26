#!/usr/bin/env python3
"""Run deterministic financial-risk inference and write independent metrics."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LABEL_RE = re.compile(r"(?im)^\s*Risk\s*:\s*(BANKRUPTCY|HEALTHY)\s*$")


def parse_label(text: str) -> int | None:
    match = LABEL_RE.search(text or "")
    if not match:
        return None
    return 1 if match.group(1).upper() == "BANKRUPTCY" else 0


def calculate_metrics(labels: list[int], predictions: list[int | None]) -> dict:
    tp = fp = tn = fn = invalid = invalid_negative = 0
    for label, prediction in zip(labels, predictions):
        if prediction is None:
            invalid += 1
            if label == 1:
                fn += 1
            else:
                invalid_negative += 1
            continue
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 0 and prediction == 1:
            fp += 1
        elif label == 0 and prediction == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
    total = len(labels)
    return {
        "samples": total,
        "positive_samples": sum(labels),
        "negative_samples": total - sum(labels),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "invalid_output": invalid,
        "invalid_negative": invalid_negative,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "false_negative_rate": fn / (tp + fn) if tp + fn else 0.0,
        "format_valid_rate": (total - invalid) / total if total else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    import pandas as pd
    from vllm import LLM, SamplingParams

    frame = pd.read_parquet(args.data)
    if args.limit:
        frame = frame.head(args.limit)
    conversations = []
    for _, row in frame.iterrows():
        raw_prompt = row["prompt"]
        conversations.append(
            raw_prompt.tolist() if hasattr(raw_prompt, "tolist") else list(raw_prompt)
        )
    labels = [int(row["extra_info"]["label"]) for _, row in frame.iterrows()]

    engine = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=False,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.75,
    )
    tokenizer = engine.get_tokenizer()
    prompts = []
    for conversation in conversations:
        prompts.append(
            tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        )

    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    outputs = engine.generate(prompts, sampling)
    responses = [output.outputs[0].text for output in outputs]
    predictions = [parse_label(response) for response in responses]
    metrics = calculate_metrics(labels, predictions)

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for (_, row), label, prediction, response in zip(frame.iterrows(), labels, predictions, responses):
        records.append({
            "sample_id": row["extra_info"]["sample_id"],
            "label": label,
            "prediction": prediction,
            "correct": prediction == label,
            "response": response,
        })
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
