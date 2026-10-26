#!/usr/bin/env python3
"""Preflight converted FinQA parquet against the target VERL model tokenizer."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_reward(path: Path):
    spec = importlib.util.spec_from_file_location("finqa_reward_external", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    args = parser.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    reward_fn = load_reward(args.reward)
    rows = []
    reward_tests = 0
    for split in ("train", "dev", "test"):
        frame = pd.read_parquet(args.data / f"{split}.parquet")
        for index, record in frame.iterrows():
            messages = list(record["prompt"])
            tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            rows.append({"split": split, "index": index, "token_count": len(tokens)})
        sample = frame.iloc[0]
        truth = json.loads(sample["reward_model"]["ground_truth"])
        response = (
            f"Answer: {truth['answer']}\n"
            f"Program: {truth['gold_program']}\n"
            f"Evidence: {', '.join(truth['gold_evidence'])}\n"
            "End: FINQA_END"
        )
        result = reward_fn(
            data_source=sample["data_source"],
            solution_str=response,
            ground_truth=sample["reward_model"]["ground_truth"],
            extra_info=sample["extra_info"],
        )
        if result["score"] != 4.0 or not result["acc"]:
            raise AssertionError({"split": split, "result": result})
        reward_tests += 1

    overlength = [row for row in rows if row["token_count"] > args.max_prompt_length]
    by_split = {}
    for split in ("train", "dev", "test"):
        values = [row["token_count"] for row in rows if row["split"] == split]
        by_split[split] = {
            "samples": len(values),
            "mean_tokens": sum(values) / len(values),
            "max_tokens": max(values),
            "overlength": sum(value > args.max_prompt_length for value in values),
        }
    report = {
        "status": "pass" if not overlength else "blocked_overlength",
        "max_prompt_length": args.max_prompt_length,
        "samples": len(rows),
        "reward_tests": reward_tests,
        "splits": by_split,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if overlength else 0


if __name__ == "__main__":
    raise SystemExit(main())
