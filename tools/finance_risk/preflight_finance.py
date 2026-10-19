#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
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
    rows = []
    for split in ("train", "test"):
        config = OmegaConf.create({
            "prompt_key": "prompt",
            "max_prompt_length": args.max_prompt_length,
            "filter_overlong_prompts": False,
            "filter_prompts": False,
            "shuffle": False,
            "cache_dir": str(args.output / "cache"),
        })
        dataset = RLHFDataset([str(args.data / f"{split}.parquet")], tokenizer, config)
        for index in range(len(dataset)):
            item = dataset[index]
            messages = list(item["raw_prompt"])
            tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            rows.append({"split": split, "index": index, "token_count": len(tokens)})
    tests = [
        ("Risk: BANKRUPTCY\nEvidence: Current Ratio=0.1; Quick Ratio=0.2", "BANKRUPTCY", 4.0),
        ("Risk: HEALTHY\nEvidence: Current Ratio=0.1; Quick Ratio=0.2", "BANKRUPTCY", -2.0),
        ("unknown", "HEALTHY", -4.0),
    ]
    for response, truth, expected in tests:
        result = reward_fn(
            data_source="finance_risk_taiwanese_bankruptcy",
            solution_str=response,
            ground_truth=truth,
            extra_info={"feature_names": ["Current Ratio", "Quick Ratio"]},
        )
        if result["score"] != expected:
            raise AssertionError((response, result, expected))
    overlength = [row for row in rows if row["token_count"] > args.max_prompt_length]
    result = {
        "status": "pass" if not overlength else "blocked_overlength",
        "samples": len(rows),
        "max_tokens": max(row["token_count"] for row in rows),
        "mean_tokens": sum(row["token_count"] for row in rows) / len(rows),
        "overlength": len(overlength),
        "reward_tests": len(tests),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if overlength:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
