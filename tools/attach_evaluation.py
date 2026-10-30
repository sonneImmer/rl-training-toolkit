#!/usr/bin/env python3
"""Attach an independent evaluation result to a dashboard run directory."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset")
    parser.add_argument("--source-model")
    args = parser.parse_args()
    if not (args.run_dir / "train.log").is_file():
        parser.error("--run-dir must contain train.log")
    try:
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"cannot read metrics JSON: {error}")
    if not isinstance(metrics, dict):
        parser.error("metrics JSON must contain an object")
    destination = args.run_dir / "evaluations" / args.evaluation_id
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.metrics, destination / "metrics.json")
    manifest = {
        "schema_version": "1.0.0", "evaluation_id": args.evaluation_id,
        "checkpoint_step": args.checkpoint_step, "split": args.split,
        "dataset": args.dataset, "source_model": args.source_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_metrics": str(args.metrics.resolve()),
    }
    (destination / "evaluation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
