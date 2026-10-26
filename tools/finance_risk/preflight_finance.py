#!/usr/bin/env python3
"""Validate financial-risk RL data, prompt lengths, and outcome reward."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    args = parser.parse_args()

    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.import_utils import load_extern_object

    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    reward_fn = load_extern_object(module_path=str(args.reward), object_name="compute_score")
    reward_tests = load_extern_object(module_path=str(args.reward), object_name="REWARD_TEST_CASES")
    rows = []
    for split in ("train", "dev", "test"):
        split_path = args.data / f"{split}.parquet"
        if not split_path.is_file():
            continue
        config = OmegaConf.create({
            "prompt_key": "prompt",
            "max_prompt_length": args.max_prompt_length,
            "filter_overlong_prompts": False,
            "filter_prompts": False,
            "shuffle": False,
            "cache_dir": str(args.output / "cache"),
        })
        dataset = RLHFDataset([str(split_path)], tokenizer, config)
        for index in range(len(dataset)):
            item = dataset[index]
            tokens = tokenizer.apply_chat_template(
                list(item["raw_prompt"]), add_generation_prompt=True, tokenize=True
            )
            rows.append({"split": split, "index": index, "token_count": len(tokens)})

    reward_rule_id = "unknown"
    for response, truth, expected in reward_tests:
        reward_result = reward_fn(
            data_source="finance_risk_taiwanese_bankruptcy",
            solution_str=response,
            ground_truth=truth,
            extra_info={"feature_names": ["Current Ratio", "Quick Ratio"]},
        )
        if reward_result["score"] != expected:
            raise AssertionError((response, reward_result, expected))
        reward_rule_id = reward_result.get("reward_rule_id", reward_rule_id)

    overlength = [row for row in rows if row["token_count"] > args.max_prompt_length]
    result = {
        "status": "pass" if not overlength else "blocked_overlength",
        "samples": len(rows),
        "max_tokens": max(row["token_count"] for row in rows),
        "mean_tokens": sum(row["token_count"] for row in rows) / len(rows),
        "overlength": len(overlength),
        "reward_rule_id": reward_rule_id,
        "reward_tests": len(reward_tests),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 1 if overlength else 0


if __name__ == "__main__":
    raise SystemExit(main())
