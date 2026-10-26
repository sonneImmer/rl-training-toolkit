#!/usr/bin/env python3
"""Create FinQA SFT parquet for learning the executable output language."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from prepare_finqa import build_prompt


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_answer(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import pandas as pd

    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    stats = {}
    for split in ("train", "dev"):
        source_path = args.source / "dataset" / f"{split}.json"
        records = json.loads(source_path.read_text(encoding="utf-8"))
        rows = []
        for record in records:
            qa = record["qa"]
            response = (
                f"Answer: {format_answer(qa['exe_ans'])}\n"
                f"Program: {qa['program']}\n"
                f"Evidence: {', '.join(sorted(qa['gold_inds'].keys()))}\n"
                "End: FINQA_END"
            )
            rows.append({
                "prompt": build_prompt(record),
                "response": response,
                "sample_id": record["id"],
            })
        output_path = args.output / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(output_path, index=False)
        stats[split] = {"rows": len(rows), "sha256": sha256(output_path)}

    manifest = {
        "schema_version": "1.0.0",
        "source": "FinQA official repository",
        "task": "financial_numerical_reasoning_sft",
        "target": "Answer/Program/Evidence/End executable format",
        "splits": stats,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
