#!/usr/bin/env python3
"""Convert official FinQA JSON files to VERL parquet datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SPLITS = ("train", "dev", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_row_text(header: list[str], row: list[str]) -> str:
    name = row[0] or "unnamed row"
    values = []
    for column, value in zip(header[1:], row[1:]):
        values.append(f"{column or 'value'}={value}")
    return f"{name}: " + "; ".join(values)


def build_prompt(record: dict) -> str:
    qa = record["qa"]
    retrieved = "\n".join(f"[{source_id}] {text}" for source_id, text in qa.get("model_input", []))
    table = record.get("table", [])
    table_lines = []
    if table:
        header = table[0]
        table_lines = [
            f"[table_{index}] {table_row_text(header, row)}"
            for index, row in enumerate(table[1:], start=1)
        ]
    context = "\n".join(filter(None, [retrieved, "\n".join(table_lines)]))
    return (
        "You are solving a numerical reasoning problem from a financial report.\n"
        "Use only the supplied evidence. Produce a sequential executable program; #0 refers to the first operation result, #1 to the second, and so on.\n"
        "Allowed operations: add, subtract, multiply, divide, exp, greater, table_max, table_min, table_sum, table_average.\n\n"
        f"Evidence:\n{context}\n\n"
        f"Question: {qa['question']}\n\n"
        "Return exactly four lines and stop after the terminator:\n"
        "Answer: final value or yes/no\n"
        "Program: operation(arg1, arg2), operation(#0, arg2)\n"
        "Evidence: comma-separated source IDs such as text_1, table_2\n"
        "End: FINQA_END\n"
        "/no_think"
    )


def convert_record(record: dict, split: str, index: int) -> dict:
    qa = record["qa"]
    ground_truth = {
        "answer": qa["exe_ans"],
        "gold_program": qa["program"],
        "gold_evidence": sorted(qa["gold_inds"].keys()),
    }
    extra_info = {
        "sample_id": record["id"],
        "split": split,
        "index": index,
        "table": json.dumps(record.get("table", []), ensure_ascii=False),
        "gold_program_steps": len(qa.get("steps", [])),
    }
    return {
        "prompt": [{"role": "user", "content": build_prompt(record)}],
        "reward_model": {"ground_truth": json.dumps(ground_truth, ensure_ascii=False)},
        "data_source": "finqa_executable_reasoning",
        "extra_info": extra_info,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Official FinQA repository root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import pandas as pd

    dataset_dir = args.source / "dataset"
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)

    split_stats = {}
    for split in SPLITS:
        source_path = dataset_dir / f"{split}.json"
        records = json.loads(source_path.read_text(encoding="utf-8"))
        converted = [convert_record(record, split, index) for index, record in enumerate(records)]
        output_path = args.output / f"{split}.parquet"
        pd.DataFrame(converted).to_parquet(output_path, index=False)
        split_stats[split] = {
            "rows": len(converted),
            "source_sha256": sha256(source_path),
            "parquet_sha256": sha256(output_path),
        }

    manifest = {
        "schema_version": "1.0.0",
        "source": "FinQA official repository",
        "source_path": str(args.source.resolve()),
        "license": "MIT",
        "task": "financial_numerical_reasoning",
        "context": "official model_input plus financial table",
        "reasoning_supervision": "executable sequential program",
        "contains_free_text_cot": False,
        "reward": {
            "definition": "finqa_execution_answer_evidence_v2",
            "min": -4.0,
            "max": 4.0,
            "primary_signal": "program execution result equals gold answer",
        },
        "splits": split_stats,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
