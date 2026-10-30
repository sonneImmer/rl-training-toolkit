"""Load independent checkpoint evaluation evidence attached to a run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _numeric_summary(metrics: dict[str, Any]) -> dict[str, float | int]:
    risk = metrics.get("risk") if isinstance(metrics.get("risk"), dict) else {}
    aliases = {
        "samples": metrics.get("samples"),
        "risk_accuracy": risk.get("accuracy"), "risk_precision": risk.get("precision"),
        "risk_recall": risk.get("recall"), "risk_f1": risk.get("f1"), "risk_f2": risk.get("f2"),
        "false_positive": risk.get("false_positive"), "false_negative": risk.get("false_negative"),
        "typology_accuracy": metrics.get("typology_accuracy"),
        "format_valid_rate": metrics.get("format_valid_rate"), "evidence_f1": metrics.get("evidence_f1"),
        "accuracy": metrics.get("accuracy"), "precision": metrics.get("precision"),
        "recall": metrics.get("recall"), "f1": metrics.get("f1"), "f2": metrics.get("f2"),
    }
    return {key: value for key, value in aliases.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}


def find_evaluation_tradeoffs(evaluations: list[dict[str, Any]], class_drop_threshold: float = 0.05) -> list[dict[str, Any]]:
    """Find overall improvement that hides a material per-class regression."""
    alerts = []
    ordered = [item for item in evaluations if item.get("checkpoint_step") is not None]
    for before, after in zip(ordered, ordered[1:]):
        before_summary, after_summary = before.get("summary", {}), after.get("summary", {})
        overall_key = next((key for key in ("risk_accuracy", "accuracy", "typology_accuracy") if key in before_summary and key in after_summary), None)
        if overall_key is None or after_summary[overall_key] <= before_summary[overall_key]:
            continue
        shared_classes = set(before.get("per_class", {})) & set(after.get("per_class", {}))
        for class_name in sorted(shared_classes):
            old = before["per_class"][class_name].get("accuracy")
            new = after["per_class"][class_name].get("accuracy")
            if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
                continue
            if old - new >= class_drop_threshold:
                alerts.append({
                    "type": "overall_up_class_down", "metric": overall_key,
                    "from_step": before["checkpoint_step"], "to_step": after["checkpoint_step"],
                    "overall_before": before_summary[overall_key], "overall_after": after_summary[overall_key],
                    "class_name": class_name, "class_before": old, "class_after": new,
                    "threshold": class_drop_threshold,
                })
    return alerts


def load_evaluations(run_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(run_dir) / "evaluations"
    if not root.is_dir():
        return []
    evaluations = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metrics, dict):
            continue
        manifest = {}
        manifest_path = metrics_path.with_name("evaluation_manifest.json")
        if manifest_path.is_file():
            try:
                value = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest = value if isinstance(value, dict) else {}
            except (OSError, json.JSONDecodeError):
                pass
        evaluations.append({
            "evaluation_id": manifest.get("evaluation_id") or metrics_path.parent.name,
            "checkpoint_step": manifest.get("checkpoint_step"),
            "split": manifest.get("split", "test"),
            "dataset": manifest.get("dataset"),
            "source_model": manifest.get("source_model"),
            "created_at": manifest.get("created_at"),
            "summary": _numeric_summary(metrics),
            "per_class": metrics.get("per_typology") or metrics.get("per_class") or {},
            "confusion": metrics.get("typology_confusion") or metrics.get("confusion_matrix") or {},
            "metrics": metrics,
            "metrics_path": str(metrics_path),
        })
    evaluations.sort(key=lambda item: (item.get("checkpoint_step") is None, item.get("checkpoint_step") or 0, item["evaluation_id"]))
    return evaluations
