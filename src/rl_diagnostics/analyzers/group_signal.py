"""REQ-03 Layer 2: Group learning signal diagnosis.

Determines whether a group of rollouts provides enough reward
distinction for GRPO-style advantage computation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..schemas.data_models import GroupRecord, RolloutRecord
from .rollout_quality import RolloutQualityResult


@dataclass
class GroupSignalResult:
    """Learning signal diagnosis result for a single group."""
    experiment_id: str
    step: int
    group_id: str
    signal_type: str  # constant_score | near_constant_score | effective_group | insufficient_group | group_unresolved
    reason: str = ""
    rollout_count: int = 0
    usable_count: int = 0
    rewards: list = field(default_factory=list)
    unique_reward_count: int = 0
    reward_range: float = 0.0
    reward_std: float = 0.0
    informative_pair_ratio: float = 0.0
    positive_reward_ratio: Optional[float] = None
    reward_outcome_type: Optional[str] = None
    low_information: bool = False
    thresholds: dict = field(default_factory=dict)
    # Correctness (only when independent verifier available)
    correctness_ratio: Optional[float] = None
    correctness_type: Optional[str] = None  # all/mostly correct, all/mostly wrong, mixed, unknown, partial
    diagnostic_reason: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class GroupSignalConfig:
    """Configuration for group signal diagnosis."""
    abs_epsilon: float = 1e-6
    rel_epsilon: float = 1e-4
    min_usable_rollouts: int = 2
    min_effective_ratio: float = 0.3  # min fraction of distinguishable pairs
    near_constant_abs_threshold: float = 1e-3
    near_constant_rel_threshold: float = 1e-2
    extreme_ratio_threshold: float = 0.75

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def _group_reward_scale(rewards: list[float]) -> float:
    """Compute a robust scale for rewards in a group.

    Uses max absolute value as scale. Falls back to 1.0 if all zeros.
    """
    if not rewards:
        return 1.0
    max_abs = max(abs(r) for r in rewards)
    if max_abs == 0:
        return 1.0
    return max_abs


def _is_distinguishable(r1: float, r2: float, abs_eps: float,
                         rel_eps: float, scale: float) -> bool:
    """Check if two rewards are distinguishable under combined tolerance."""
    diff = abs(r1 - r2)
    threshold = max(abs_eps, rel_eps * scale)
    return diff > threshold


def _count_informative_pairs(rewards: list[float], config: GroupSignalConfig) -> tuple[int, int]:
    """Count distinguishable pairs and total pairs.
    
    Returns (informative_pairs, total_pairs).
    """
    n = len(rewards)
    if n < 2:
        return 0, 0

    scale = _group_reward_scale(rewards)
    total = n * (n - 1) // 2
    informative = 0

    for i in range(n):
        for j in range(i + 1, n):
            if _is_distinguishable(rewards[i], rewards[j],
                                    config.abs_epsilon, config.rel_epsilon, scale):
                informative += 1

    return informative, total


def _count_unique_rewards(rewards: list[float], config: GroupSignalConfig) -> int:
    """Count unique rewards under tolerance."""
    if not rewards:
        return 0
    scale = _group_reward_scale(rewards)
    threshold = max(config.abs_epsilon, config.rel_epsilon * scale)

    unique = []
    for r in rewards:
        is_new = True
        for u in unique:
            if abs(r - u) <= threshold:
                is_new = False
                break
        if is_new:
            unique.append(r)
    return len(unique)


def diagnose_group_signal(
    group: GroupRecord,
    quality_results: list[RolloutQualityResult],
    config: Optional[GroupSignalConfig] = None,
    correctness_labels: Optional[list[Optional[bool]]] = None,
) -> GroupSignalResult:
    """Diagnose learning signal for a single group.
    
    Args:
        group: The standardized group record.
        quality_results: Quality results for rollouts in this group.
        config: Diagnostic configuration. Uses defaults if None.
        correctness_labels: Optional list of independent correctness labels
                           (True/False/None) for each rollout.
    
    Returns:
        GroupSignalResult with signal type and metrics.
    """
    if config is None:
        config = GroupSignalConfig()

    exp_id = group.experiment_id
    step = group.step
    gid = group.group_id

    # Filter to usable rollouts
    usable_results = [q for q in quality_results
                      if q.group_id == gid and q.step == step
                      and q.status == "usable"]
    usable_count = len(usable_results)

    # Extract usable rewards
    usable_rewards = []
    for q in usable_results:
        # Find matching rollout in group
        for i, rid in enumerate(group.rollout_ids):
            if rid == q.rollout_id and i < len(group.rewards):
                usable_rewards.append(group.rewards[i])
                break

    # Also try direct reward list
    if not usable_rewards and group.rewards:
        usable_rewards = [r for r in group.rewards if r is not None and _is_finite(r)]

    # Check group resolvability
    if group.group_status == "group_unresolved":
        return GroupSignalResult(
            experiment_id=exp_id, step=step, group_id=gid,
            signal_type="group_unresolved",
            reason="Group structure could not be recovered",
            rollout_count=group.rollout_count,
            usable_count=usable_count,
            rewards=usable_rewards,
            thresholds=config.to_dict(),
            diagnostic_reason="Group structure unrecoverable; no group-level statistics produced",
        )

    # Layer 1: Insufficient rollouts
    if usable_count < config.min_usable_rollouts:
        return GroupSignalResult(
            experiment_id=exp_id, step=step, group_id=gid,
            signal_type="insufficient_group",
            reason=f"Only {usable_count} usable rollouts (min={config.min_usable_rollouts})",
            rollout_count=group.rollout_count,
            usable_count=usable_count,
            rewards=usable_rewards,
            thresholds=config.to_dict(),
            diagnostic_reason=f"Insufficient usable rollouts for reliable group analysis",
        )

    # Compute metrics
    n = len(usable_rewards)
    unique_count = _count_unique_rewards(usable_rewards, config)
    informative_pairs, total_pairs = _count_informative_pairs(usable_rewards, config)
    informative_ratio = informative_pairs / total_pairs if total_pairs > 0 else 0.0

    reward_range = max(usable_rewards) - min(usable_rewards) if usable_rewards else 0.0
    reward_mean = sum(usable_rewards) / n if n > 0 else 0.0
    reward_std = math.sqrt(sum((r - reward_mean) ** 2 for r in usable_rewards) / n) if n > 0 else 0.0
    positive_reward_ratio = sum(r > 0 for r in usable_rewards) / n if n > 0 else None
    reward_outcome_type = _classify_ratio(
        positive_reward_ratio, config.extreme_ratio_threshold,
        "positive", "non_positive", "mixed_reward",
    )

    # Layer 2: Constant score (all same within tolerance)
    if unique_count <= 1:
        signal_type = "constant_score"
        reason = f"All {n} rewards identical within tolerance"
        low_info = True
    # Layer 3: Near constant (some difference but below threshold)
    elif (reward_range <= config.near_constant_abs_threshold or
          reward_range / _group_reward_scale(usable_rewards) <= config.near_constant_rel_threshold):
        signal_type = "near_constant_score"
        reason = f"Reward range {reward_range:.6f} below near-constant threshold"
        low_info = True
    # Layer 4: Check if effective
    elif informative_ratio >= config.min_effective_ratio:
        signal_type = "effective_group"
        reason = f"{informative_pairs}/{total_pairs} distinguishable pairs (ratio={informative_ratio:.3f})"
        low_info = False
    else:
        signal_type = "near_constant_score"
        reason = f"Low informative pair ratio {informative_ratio:.3f} (min={config.min_effective_ratio})"
        low_info = True

    # Correctness (only with independent labels)
    correctness_type = None
    correctness_ratio = None
    if correctness_labels is not None:
        valid_labels = [l for l in correctness_labels if l is not None]
        if len(valid_labels) == len(correctness_labels) and len(valid_labels) > 0:
            correctness_ratio = sum(valid_labels) / len(valid_labels)
            correctness_type = _classify_ratio(
                correctness_ratio, config.extreme_ratio_threshold,
                "correct", "wrong", "mixed_correctness",
            )
        elif len(valid_labels) > 0:
            correctness_type = "partial"
        else:
            correctness_type = "unknown"

    # Build diagnostic reason
    diag_reason = f"signal_type={signal_type}; "
    diag_reason += f"usable={usable_count}/{group.rollout_count}; "
    diag_reason += f"unique_rewards={unique_count}; "
    diag_reason += f"informative_ratio={informative_ratio:.4f}; "
    diag_reason += f"reward_range={reward_range:.6f}; "
    diag_reason += f"reward_std={reward_std:.6f}"
    if correctness_type is not None:
        diag_reason += f"; correctness={correctness_type}"

    return GroupSignalResult(
        experiment_id=exp_id, step=step, group_id=gid,
        signal_type=signal_type,
        reason=reason,
        rollout_count=group.rollout_count,
        usable_count=usable_count,
        rewards=usable_rewards,
        unique_reward_count=unique_count,
        reward_range=reward_range,
        reward_std=reward_std,
        informative_pair_ratio=informative_ratio,
        positive_reward_ratio=positive_reward_ratio,
        reward_outcome_type=reward_outcome_type,
        low_information=low_info,
        thresholds=config.to_dict(),
        correctness_ratio=correctness_ratio,
        correctness_type=correctness_type,
        diagnostic_reason=diag_reason,
    )


def _is_finite(val) -> bool:
    """Check if value is finite number."""
    if val is None:
        return False
    try:
        v = float(val)
        return not (math.isnan(v) or math.isinf(v))
    except (TypeError, ValueError):
        return False


def _classify_ratio(
    ratio: Optional[float], threshold: float,
    positive_label: str, negative_label: str, mixed_label: str,
) -> Optional[str]:
    if ratio is None:
        return None
    if ratio == 1.0:
        return f"all_{positive_label}"
    if ratio == 0.0:
        return f"all_{negative_label}"
    if ratio >= threshold:
        return f"mostly_{positive_label}"
    if ratio <= 1.0 - threshold:
        return f"mostly_{negative_label}"
    return mixed_label


@dataclass
class StepAggregation:
    """Aggregated metrics for a single training step."""
    step: int
    epoch: Optional[int]
    total_groups: int = 0
    analyzable_groups: int = 0
    effective_groups: int = 0
    constant_score_groups: int = 0
    near_constant_groups: int = 0
    insufficient_groups: int = 0
    unresolved_groups: int = 0
    total_rollouts: int = 0
    usable_rollouts: int = 0
    rollout_usable_ratio: float = 0.0
    effective_group_ratio: float = 0.0
    low_information_group_ratio: float = 0.0
    constant_score_group_ratio: float = 0.0
    near_constant_group_ratio: float = 0.0
    all_correct_group_ratio: Optional[float] = None
    all_wrong_group_ratio: Optional[float] = None
    mostly_correct_group_ratio: Optional[float] = None
    mostly_wrong_group_ratio: Optional[float] = None
    mean_informative_pair_ratio: float = 0.0

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def aggregate_step(
    step: int,
    epoch: Optional[int],
    group_results: list[GroupSignalResult],
    quality_results: list[RolloutQualityResult],
) -> StepAggregation:
    """Aggregate group signal results for a single step."""
    step_groups = [g for g in group_results if g.step == step]
    step_quality = [q for q in quality_results if q.step == step]

    total_groups = len(step_groups)
    analyzable = [g for g in step_groups if g.signal_type not in ("group_unresolved",)]
    analyzable_count = len(analyzable)

    effective = sum(1 for g in step_groups if g.signal_type == "effective_group")
    constant = sum(1 for g in step_groups if g.signal_type == "constant_score")
    near_const = sum(1 for g in step_groups if g.signal_type == "near_constant_score")
    insufficient = sum(1 for g in step_groups if g.signal_type == "insufficient_group")
    unresolved = sum(1 for g in step_groups if g.signal_type == "group_unresolved")

    total_rollouts = len(step_quality)
    usable_rollouts = sum(1 for q in step_quality if q.status == "usable")

    # Ratios
    usable_ratio = usable_rollouts / total_rollouts if total_rollouts > 0 else 0.0
    eff_ratio = effective / analyzable_count if analyzable_count > 0 else 0.0
    low_info_ratio = (constant + near_const) / analyzable_count if analyzable_count > 0 else 0.0
    const_ratio = constant / analyzable_count if analyzable_count > 0 else 0.0
    near_const_ratio = near_const / analyzable_count if analyzable_count > 0 else 0.0

    # Correctness ratios (only if available)
    all_correct = sum(1 for g in step_groups if g.correctness_type == "all_correct")
    all_wrong = sum(1 for g in step_groups if g.correctness_type == "all_wrong")
    mostly_correct = sum(1 for g in step_groups if g.correctness_type == "mostly_correct")
    mostly_wrong = sum(1 for g in step_groups if g.correctness_type == "mostly_wrong")
    has_correctness = any(g.correctness_type is not None for g in step_groups)

    all_correct_ratio = all_correct / analyzable_count if (analyzable_count > 0 and has_correctness) else None
    all_wrong_ratio = all_wrong / analyzable_count if (analyzable_count > 0 and has_correctness) else None
    mostly_correct_ratio = mostly_correct / analyzable_count if (analyzable_count > 0 and has_correctness) else None
    mostly_wrong_ratio = mostly_wrong / analyzable_count if (analyzable_count > 0 and has_correctness) else None

    # Mean informative pair ratio
    informative_ratios = [g.informative_pair_ratio for g in analyzable if g.signal_type != "insufficient_group"]
    mean_ipr = sum(informative_ratios) / len(informative_ratios) if informative_ratios else 0.0

    return StepAggregation(
        step=step, epoch=epoch,
        total_groups=total_groups,
        analyzable_groups=analyzable_count,
        effective_groups=effective,
        constant_score_groups=constant,
        near_constant_groups=near_const,
        insufficient_groups=insufficient,
        unresolved_groups=unresolved,
        total_rollouts=total_rollouts,
        usable_rollouts=usable_rollouts,
        rollout_usable_ratio=usable_ratio,
        effective_group_ratio=eff_ratio,
        low_information_group_ratio=low_info_ratio,
        constant_score_group_ratio=const_ratio,
        near_constant_group_ratio=near_const_ratio,
        all_correct_group_ratio=all_correct_ratio,
        all_wrong_group_ratio=all_wrong_ratio,
        mostly_correct_group_ratio=mostly_correct_ratio,
        mostly_wrong_group_ratio=mostly_wrong_ratio,
        mean_informative_pair_ratio=mean_ipr,
    )
