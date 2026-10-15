"""REQ-03 Layer 1: Rollout data quality assessment.

Determines whether each rollout is structurally usable for downstream analysis.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from ..schemas.data_models import RolloutRecord

# Template-only patterns that indicate empty content
_TEMPLATE_PATTERNS = [
    re.compile(r"^[\s\n]*$"),  # whitespace only
    re.compile(r"^<{2,}.*>{2,}$"),  # placeholder like <<answer>>
    re.compile(r"^\[.*placeholder.*\]$", re.IGNORECASE),
    re.compile(r"^\s*(none|null|n/a|empty)\s*$", re.IGNORECASE),
]


@dataclass
class RolloutQualityResult:
    """Quality assessment result for a single rollout."""
    experiment_id: str
    step: int
    group_id: str
    rollout_id: int
    status: str  # usable | invalid_empty | invalid_reward | invalid_format | invalid_truncated | invalid_verifier
    reason: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def _is_empty_response(response: Optional[str]) -> bool:
    """Check if response is empty or only template characters."""
    if response is None:
        return True
    text = response.strip()
    if not text:
        return True
    for pattern in _TEMPLATE_PATTERNS:
        if pattern.match(text):
            return True
    return False


def _is_finite_reward(reward: Optional[float]) -> bool:
    """Check if reward is a finite number."""
    if reward is None:
        return False
    if isinstance(reward, bool):
        return False
    try:
        val = float(reward)
        return not (math.isnan(val) or math.isinf(val))
    except (TypeError, ValueError):
        return False


def _is_truncated(response: Optional[str], finish_reason: Optional[str],
                  max_response_length: Optional[int],
                  response_token_length: Optional[int]) -> bool:
    """Check if response was truncated."""
    if finish_reason and finish_reason in ("length", "max_tokens"):
        return True
    if max_response_length is not None and response_token_length is not None:
        if response_token_length >= max_response_length:
            return True
    return False


def assess_rollout_quality(
    rollout: RolloutRecord,
    format_checker=None,
    verifier_result: Optional[dict] = None,
) -> RolloutQualityResult:
    """Assess a single rollout's data quality.
    
    Args:
        rollout: The standardized rollout record.
        format_checker: Optional callable(str) -> (bool, str) that checks
                        response format. Returns (is_valid, reason).
        verifier_result: Optional dict with verifier status.
    
    Returns:
        RolloutQualityResult with status and reason.
    """
    exp_id = rollout.experiment_id
    step = rollout.step
    gid = rollout.group_id
    rid = rollout.rollout_id

    # Check parse status first
    if rollout.parse_status == "json_error":
        return RolloutQualityResult(
            experiment_id=exp_id, step=step, group_id=gid, rollout_id=rid,
            status="invalid_reward", reason=f"JSON parse error: {rollout.parse_error}",
        )
    if rollout.parse_status == "schema_error":
        return RolloutQualityResult(
            experiment_id=exp_id, step=step, group_id=gid, rollout_id=rid,
            status="invalid_reward", reason=f"Schema error: {rollout.parse_error}",
        )

    # Layer 1: Empty response
    if _is_empty_response(rollout.response):
        return RolloutQualityResult(
            experiment_id=exp_id, step=step, group_id=gid, rollout_id=rid,
            status="invalid_empty", reason="Response is empty or only template characters",
        )

    # Layer 2: Invalid reward
    if not _is_finite_reward(rollout.training_reward):
        reason = "missing"
        if rollout.training_reward is not None:
            val = rollout.training_reward
            if isinstance(val, float) and math.isnan(val):
                reason = "NaN"
            elif isinstance(val, float) and math.isinf(val):
                reason = "Inf"
            else:
                reason = f"non-numeric: {val}"
        return RolloutQualityResult(
            experiment_id=exp_id, step=step, group_id=gid, rollout_id=rid,
            status="invalid_reward", reason=f"Reward is {reason}",
        )

    # Layer 3: Truncated
    if _is_truncated(rollout.response, rollout.finish_reason,
                      rollout.max_response_length, rollout.response_token_length):
        return RolloutQualityResult(
            experiment_id=exp_id, step=step, group_id=gid, rollout_id=rid,
            status="invalid_truncated",
            reason="Response reached max length without completion",
        )

    # Layer 4: Format check (optional)
    if format_checker is not None:
        try:
            is_valid, fmt_reason = format_checker(rollout.response)
            if not is_valid:
                return RolloutQualityResult(
                    experiment_id=exp_id, step=step, group_id=gid, rollout_id=rid,
                    status="invalid_format", reason=fmt_reason,
                )
        except Exception as e:
            return RolloutQualityResult(
                experiment_id=exp_id, step=step, group_id=gid, rollout_id=rid,
                status="invalid_format", reason=f"Format checker error: {e}",
            )

    # Layer 5: Verifier failure (optional)
    if verifier_result is not None:
        v_status = verifier_result.get("status", "not_available")
        if v_status in ("execution_error", "timeout"):
            return RolloutQualityResult(
                experiment_id=exp_id, step=step, group_id=gid, rollout_id=rid,
                status="invalid_verifier",
                reason=f"Verifier {v_status}: {verifier_result.get('error_type', 'unknown')}",
            )

    return RolloutQualityResult(
        experiment_id=exp_id, step=step, group_id=gid, rollout_id=rid,
        status="usable", reason="",
    )
