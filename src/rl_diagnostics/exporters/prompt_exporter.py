"""REQ-05: Abnormal prompt exporter.

Exports prompts that triggered diagnostic warnings for manual review,
data cleaning, or resampling.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ..schemas.data_models import GroupRecord, RolloutRecord
from ..analyzers.rollout_quality import RolloutQualityResult
from ..analyzers.group_signal import GroupSignalResult


@dataclass
class ExportFilter:
    """Filter criteria for prompt export."""
    experiment_id: Optional[str] = None
    steps: Optional[list[int]] = None
    epochs: Optional[list[int]] = None
    diagnosis_types: Optional[list[str]] = None  # constant_score, near_constant_score, invalid_empty, etc.
    difficulties: Optional[list[str]] = None
    source_datasets: Optional[list[str]] = None

    def matches(self, step: int, epoch: Optional[int], diag_type: str,
                difficulty: Optional[str], source_ds: Optional[str],
                exp_id: str) -> bool:
        if self.experiment_id and exp_id != self.experiment_id:
            return False
        if self.steps and step not in self.steps:
            return False
        if self.epochs and epoch is not None and epoch not in self.epochs:
            return False
        if self.diagnosis_types and diag_type not in self.diagnosis_types:
            return False
        if self.difficulties and difficulty and difficulty not in self.difficulties:
            return False
        if self.source_datasets and source_ds and source_ds not in self.source_datasets:
            return False
        return True


def _determine_suggested_usage(diag_type: str) -> list[str]:
    """Determine suggested usage based on diagnosis type."""
    usage = []
    if diag_type in ("constant_score", "near_constant_score"):
        usage.extend(["manual_review", "resampling"])
    if diag_type in ("invalid_empty", "invalid_reward", "invalid_format", "invalid_truncated"):
        usage.extend(["data_cleaning", "manual_review"])
    if diag_type == "insufficient_group":
        usage.extend(["resampling"])
    if diag_type == "group_unresolved":
        usage.extend(["manual_review"])
    if not usage:
        usage = ["manual_review"]
    return list(dict.fromkeys(usage))  # deduplicate preserving order


def _determine_diagnosis_type(
    group_signal: Optional[GroupSignalResult],
    quality_results: list[RolloutQualityResult],
) -> str:
    """Determine the primary diagnosis type for a group."""
    # Priority: group signal type > quality issues
    if group_signal is not None:
        if group_signal.signal_type in ("constant_score", "near_constant_score",
                                          "insufficient_group", "group_unresolved"):
            return group_signal.signal_type

    # Check for quality issues
    for q in quality_results:
        if q.status != "usable":
            return q.status

    if group_signal is not None and group_signal.signal_type == "effective_group":
        return "effective_group"

    return "unknown"


def export_prompts(
    groups: list[GroupRecord],
    rollouts: list[RolloutRecord],
    quality_results: list[RolloutQualityResult],
    group_signals: list[GroupSignalResult],
    output_path: str,
    export_filter: Optional[ExportFilter] = None,
    deduplicate: bool = True,
    rule_version: str = "1.0.0",
) -> tuple[int, int, list[dict]]:
    """Export abnormal prompts to JSONL.
    
    Args:
        groups: All group records.
        rollouts: All rollout records.
        quality_results: All quality assessment results.
        group_signals: All group signal diagnosis results.
        output_path: Path to output JSONL file.
        export_filter: Optional filter criteria.
        deduplicate: If True, deduplicate by prompt_id.
        rule_version: Rule version string.
    
    Returns:
        Tuple of (exported_count, blocked_count, blocked_records).
    """
    if export_filter is None:
        export_filter = ExportFilter()

    # Index quality results by (group_id, step)
    quality_by_group: dict[tuple[str, int], list[RolloutQualityResult]] = {}
    for q in quality_results:
        key = (q.group_id, q.step)
        quality_by_group.setdefault(key, []).append(q)

    # Index group signals by (group_id, step)
    signal_by_group: dict[tuple[str, int], GroupSignalResult] = {}
    for gs in group_signals:
        signal_by_group[(gs.group_id, gs.step)] = gs

    # Index rollouts by (group_id, step)
    rollouts_by_group: dict[tuple[str, int], list[RolloutRecord]] = {}
    for r in rollouts:
        key = (r.group_id, r.step)
        rollouts_by_group.setdefault(key, []).append(r)

    exported: list[dict] = []
    blocked: list[dict] = []
    seen_prompt_ids: set[str] = set()

    for group in groups:
        key = (group.group_id, group.step)
        group_qs = quality_by_group.get(key, [])
        group_gs = signal_by_group.get(key)
        group_rollouts = rollouts_by_group.get(key, [])

        # Determine diagnosis type
        diag_type = _determine_diagnosis_type(group_gs, group_qs)

        # Skip effective groups (no abnormality)
        if diag_type == "effective_group":
            continue

        # Check if prompt can be recovered
        prompt_text = group.prompt
        prompt_id = group.prompt_id

        # Apply filter
        if not export_filter.matches(
            step=group.step,
            epoch=group.epoch,
            diag_type=diag_type,
            difficulty=group.difficulty,
            source_ds=group.source_dataset,
            exp_id=group.experiment_id,
        ):
            continue

        # Block if prompt not recoverable
        if not prompt_text or prompt_text == "None":
            blocked.append({
                "group_id": group.group_id,
                "step": group.step,
                "reason": "Prompt text not recoverable",
                "diag_type": diag_type,
            })
            continue

        # Deduplicate
        if deduplicate and prompt_id and prompt_id in seen_prompt_ids:
            # Still record triggered step
            for existing in exported:
                if existing.get("prompt_id") == prompt_id:
                    existing["triggered_steps"].append(group.step)
                    if group.group_id not in existing["group_ids"]:
                        existing["group_ids"].append(group.group_id)
                    break
            continue

        if deduplicate and prompt_id:
            seen_prompt_ids.add(prompt_id)

        # Collect responses and rewards
        responses = [r.response for r in group_rollouts]
        rewards = [r.training_reward for r in group_rollouts]

        # Build diagnostic reason
        diag_reason = ""
        if group_gs:
            diag_reason = group_gs.diagnostic_reason
        elif group_qs:
            reasons = [f"{q.status}: {q.reason}" for q in group_qs if q.status != "usable"]
            diag_reason = "; ".join(reasons) if reasons else ""

        entry = {
            "experiment_id": group.experiment_id,
            "step": group.step,
            "epoch": group.epoch,
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "source_dataset": group.source_dataset,
            "difficulty": group.difficulty,
            "triggered_steps": [group.step],
            "group_ids": [group.group_id],
            "responses": responses,
            "rewards": rewards,
            "diagnostic_type": diag_type,
            "diagnostic_reason": diag_reason,
            "rule_version": rule_version,
            "suggested_usage": _determine_suggested_usage(diag_type),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

        exported.append(entry)

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in exported:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    return len(exported), len(blocked), blocked
