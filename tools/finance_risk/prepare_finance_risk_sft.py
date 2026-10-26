#!/usr/bin/env python3
"""Build format-oriented SFT data from the fixed financial-risk training split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response_for(row) -> str:
    label = row["reward_model"]["ground_truth"]
    extra = row["extra_info"]
    values = extra["feature_values"]
    feature_names = extra["feature_names"][:2]
    evidence = "; ".join(f"{name}={values[name]:.6g}" for name in feature_names)
    return f"Risk: {label}\nEvidence: {evidence}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)

    stats = {}
    for source_split, target_split in (("train", "train"), ("dev", "dev")):
        frame = pd.read_parquet(args.source / f"{source_split}.parquet")
        output = pd.DataFrame({
            "prompt": [row["prompt"][0]["content"] for _, row in frame.iterrows()],
            "response": [response_for(row) for _, row in frame.iterrows()],
            "sample_id": [row["extra_info"]["sample_id"] for _, row in frame.iterrows()],
        })
        path = args.output / f"{target_split}.parquet"
        output.to_parquet(path, index=False)
        stats[target_split] = {"rows": len(output), "sha256": sha256(path)}

    manifest = {
        "schema_version": "1.0.0",
        "source": str(args.source),
        "task": "binary_bankruptcy_risk_format_sft",
        "target": "Risk label and input-grounded evidence format",
        "evidence_note": "Evidence fields are copied from input for format grounding; they are not expert causal explanations.",
        "splits": stats,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
