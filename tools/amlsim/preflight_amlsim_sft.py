#!/usr/bin/env python3
"""Check AMLSim SFT prompt and response token lengths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    report = {"max_length": args.max_length, "splits": {}}
    blocked = False
    for split in ("train", "dev"):
        frame = pd.read_parquet(args.data / f"{split}.parquet")
        lengths = []
        for _, row in frame.iterrows():
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}], add_generation_prompt=True, tokenize=True
            )
            response = tokenizer(row["response"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            lengths.append(len(prompt) + len(response))
        overlength = sum(length > args.max_length for length in lengths)
        blocked = blocked or overlength > 0
        report["splits"][split] = {
            "samples": len(lengths),
            "mean_tokens": sum(lengths) / len(lengths),
            "max_tokens": max(lengths),
            "overlength": overlength,
        }
    report["status"] = "blocked_overlength" if blocked else "pass"
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
