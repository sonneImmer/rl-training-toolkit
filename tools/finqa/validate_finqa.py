#!/usr/bin/env python3
"""Validate official FinQA gold programs with the local dependency-free executor."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from finqa_program import execute_program, values_equal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    failures = []
    operations = Counter()
    totals = Counter()
    for split in ("train", "dev", "test"):
        records = json.loads((args.source / "dataset" / f"{split}.json").read_text(encoding="utf-8"))
        totals[split] = len(records)
        for record in records:
            program = record["qa"]["program"]
            for call in program.split("),"):
                operations[call.strip().split("(", 1)[0]] += 1
            result = execute_program(program, record.get("table", []))
            expected = record["qa"]["exe_ans"]
            if not result.valid or not values_equal(result.value, expected):
                failures.append({
                    "id": record["id"],
                    "split": split,
                    "program": program,
                    "expected": expected,
                    "actual": result.value,
                    "error": result.error,
                })

    report = {
        "status": "pass" if not failures else "failed",
        "totals": dict(totals),
        "validated": sum(totals.values()),
        "failures": len(failures),
        "operations": dict(sorted(operations.items())),
        "failure_examples": failures[:20],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
