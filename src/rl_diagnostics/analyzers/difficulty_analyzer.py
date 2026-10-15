"""REQ-06: Optional difficulty integration.

Provides difficulty-based stratified analysis. Difficulty is optional;
when absent, analysis skips difficulty dimension gracefully.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from ..schemas.data_models import GroupRecord, RolloutRecord
from ..analyzers.group_signal import GroupSignalResult, GroupSignalConfig
from ..analyzers.rollout_quality import RolloutQualityResult


@dataclass
class DifficultyStats:
    """Aggregated stats for a single difficulty bucket."""
    difficulty: str
    group_count: int = 0
    effective_group_ratio: float = 0.0
    constant_score_group_ratio: float = 0.0
    near_constant_group_ratio: float = 0.0
    informative_pair_ratio_mean: float = 0.0
    reward_mean: float = 0.0
    reward_std: float = 0.0
    reward_range_mean: float = 0.0

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class DifficultyReport:
    """Full difficulty stratification report."""
    has_difficulty: bool = False
    missing_rate: float = 0.0
    total_groups: int = 0
    buckets: dict[str, DifficultyStats] = field(default_factory=dict)
    conflict_count: int = 0
    conflicts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "has_difficulty": self.has_difficulty,
            "missing_rate": self.missing_rate,
            "total_groups": self.total_groups,
            "buckets": {k: v.to_dict() for k, v in self.buckets.items()},
            "conflict_count": self.conflict_count,
            "conflicts": self.conflicts,
        }


def load_difficulty_map(map_path: str) -> tuple[dict[str, str], list[dict]]:
    """Load external prompt_id -> difficulty mapping from JSON file.
    
    Returns (mapping_dict, conflicts_list).
    """
    with open(map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    mapping: dict[str, str] = {}
    conflicts: list[dict] = []

    if isinstance(raw, dict):
        for prompt_id, difficulty in raw.items():
            if prompt_id in mapping and mapping[prompt_id] != difficulty:
                conflicts.append({
                    "prompt_id": prompt_id,
                    "existing": mapping[prompt_id],
                    "new": difficulty,
                })
            mapping[prompt_id] = str(difficulty)
    elif isinstance(raw, list):
        for item in raw:
            pid = item.get("prompt_id", "")
            diff = item.get("difficulty", "unknown")
            if pid in mapping and mapping[pid] != diff:
                conflicts.append({
                    "prompt_id": pid,
                    "existing": mapping[pid],
                    "new": diff,
                })
            mapping[pid] = str(diff)

    return mapping, conflicts


def assign_difficulty(
    groups: list[GroupRecord],
    external_map: Optional[dict[str, str]] = None,
) -> tuple[list[GroupRecord], DifficultyReport]:
    """Assign difficulty to groups and generate report.
    
    Priority:
    1. Group's own difficulty field (from dataset)
    2. External mapping by prompt_id
    3. "unknown" if neither available
    
    Args:
        groups: List of group records.
        external_map: Optional dict mapping prompt_id -> difficulty.
    
    Returns:
        Tuple of (updated_groups, difficulty_report).
    """
    report = DifficultyReport()
    
    if external_map is None:
        external_map = {}

    # Check for conflicts in external map
    # (already handled in load_difficulty_map, but check for one-to-many)
    seen: dict[str, list[str]] = {}
    for pid, diff in external_map.items():
        seen.setdefault(pid, []).append(diff)
    for pid, diffs in seen.items():
        if len(set(diffs)) > 1:
            report.conflicts.append({
                "prompt_id": pid,
                "difficulties": list(set(diffs)),
                "type": "one_to_many",
            })
    report.conflict_count = len(report.conflicts)

    # Assign difficulty
    updated_groups = []
    has_any_difficulty = False
    missing_count = 0

    for group in groups:
        difficulty = group.difficulty

        # Try external map if group has no difficulty
        if (difficulty is None or difficulty == "") and group.prompt_id:
            if group.prompt_id in external_map:
                difficulty = external_map[group.prompt_id]

        # Default to unknown
        if difficulty is None or difficulty == "":
            difficulty = "unknown"
            missing_count += 1
        else:
            has_any_difficulty = True

        # Create updated group
        updated = GroupRecord(
            schema_version=group.schema_version,
            experiment_id=group.experiment_id,
            step=group.step,
            epoch=group.epoch,
            generation_attempt=group.generation_attempt,
            group_id=group.group_id,
            prompt_id=group.prompt_id,
            prompt=group.prompt,
            source_dataset=group.source_dataset,
            difficulty=difficulty,
            ground_truth=group.ground_truth,
            rollout_count=group.rollout_count,
            rewards=group.rewards,
            responses=group.responses,
            rollout_ids=group.rollout_ids,
            source_file=group.source_file,
            group_status=group.group_status,
        )
        updated_groups.append(updated)

    report.has_difficulty = has_any_difficulty
    report.total_groups = len(updated_groups)
    report.missing_rate = missing_count / len(updated_groups) if updated_groups else 0.0

    return updated_groups, report


def stratify_by_difficulty(
    groups: list[GroupRecord],
    group_signals: list[GroupSignalResult],
) -> dict[str, DifficultyStats]:
    """Generate difficulty-stratified statistics.
    
    Args:
        groups: Groups with difficulty assigned.
        group_signals: Corresponding signal results.
    
    Returns:
        Dict mapping difficulty label to DifficultyStats.
    """
    # Index signals
    signal_map: dict[tuple[str, int], GroupSignalResult] = {}
    for gs in group_signals:
        signal_map[(gs.group_id, gs.step)] = gs

    buckets: dict[str, list] = {}  # difficulty -> list of (group, signal)

    for group in groups:
        diff = group.difficulty or "unknown"
        gs = signal_map.get((group.group_id, group.step))
        buckets.setdefault(diff, []).append((group, gs))

    result: dict[str, DifficultyStats] = {}

    for diff, items in buckets.items():
        total = len(items)
        effective = 0
        constant = 0
        near_const = 0
        ipr_values = []
        reward_means = []
        reward_stds = []
        reward_ranges = []

        for group, gs in items:
            if gs:
                if gs.signal_type == "effective_group":
                    effective += 1
                elif gs.signal_type == "constant_score":
                    constant += 1
                elif gs.signal_type == "near_constant_score":
                    near_const += 1
                if gs.informative_pair_ratio is not None:
                    ipr_values.append(gs.informative_pair_ratio)
                if gs.rewards:
                    reward_means.append(sum(gs.rewards) / len(gs.rewards))
                    reward_stds.append(gs.reward_std)
                    reward_ranges.append(gs.reward_range)

        analyzable = total  # all groups in bucket are analyzable
        result[diff] = DifficultyStats(
            difficulty=diff,
            group_count=total,
            effective_group_ratio=effective / analyzable if analyzable > 0 else 0.0,
            constant_score_group_ratio=constant / analyzable if analyzable > 0 else 0.0,
            near_constant_group_ratio=near_const / analyzable if analyzable > 0 else 0.0,
            informative_pair_ratio_mean=sum(ipr_values) / len(ipr_values) if ipr_values else 0.0,
            reward_mean=sum(reward_means) / len(reward_means) if reward_means else 0.0,
            reward_std=sum(reward_stds) / len(reward_stds) if reward_stds else 0.0,
            reward_range_mean=sum(reward_ranges) / len(reward_ranges) if reward_ranges else 0.0,
        )

    return result
