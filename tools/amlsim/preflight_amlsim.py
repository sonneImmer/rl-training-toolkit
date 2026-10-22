#!/usr/bin/env python3
"""Validate AMLSim RL data, prompt lengths, schema, and reward contract."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_symbol(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def as_conversation(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    args = parser.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    reward_fn = load_symbol(args.reward, "compute_score")
    reward_tests = load_symbol(args.reward, "REWARD_TEST_CASES")
    rows = []
    schema_errors = []
    sample_ids = set()
    for split in ("train", "dev", "test"):
        split_path = args.data / f"{split}.parquet"
        if not split_path.is_file():
            continue
        frame = pd.read_parquet(split_path)
        required = {"prompt", "reward_model", "data_source", "extra_info"}
        missing = sorted(required - set(frame.columns))
        if missing:
            schema_errors.append({"split": split, "missing_columns": missing})
            continue
        for index, row in frame.iterrows():
            conversation = as_conversation(row["prompt"])
            tokens = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True)
            sample_id = row["extra_info"]["sample_id"]
            if sample_id in sample_ids:
                schema_errors.append({"split": split, "index": int(index), "duplicate_sample_id": sample_id})
            sample_ids.add(sample_id)
            json.loads(row["reward_model"]["ground_truth"])
            rows.append({"split": split, "index": int(index), "token_count": len(tokens)})

    reward_rule_id = "unknown"
    for response, truth, expected in reward_tests:
        reward_result = reward_fn(data_source="amlsim_typology_rules_v1", solution_str=response, ground_truth=truth)
        if reward_result["score"] != expected:
            raise AssertionError((response, reward_result, expected))
        reward_rule_id = reward_result.get("reward_rule_id", reward_rule_id)

    overlength = [row for row in rows if row["token_count"] > args.max_prompt_length]
    blocked = bool(overlength or schema_errors or not rows)
    result = {
        "status": "blocked" if blocked else "pass",
        "samples": len(rows),
        "max_tokens": max((row["token_count"] for row in rows), default=0),
        "mean_tokens": sum(row["token_count"] for row in rows) / len(rows) if rows else 0.0,
        "overlength": len(overlength),
        "schema_errors": schema_errors,
        "reward_rule_id": reward_rule_id,
        "reward_tests": len(reward_tests),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
