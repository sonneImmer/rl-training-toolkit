"""Pure outcome reward for binary financial-risk classification.

This scorer deliberately excludes format and evidence bonuses.  Formatting and
verification details are returned as metadata for diagnostics, while the
training score is only 1 for a correct label and 0 otherwise.
"""
from __future__ import annotations

import re
from typing import Any


REWARD_TEST_CASES = [
    ("Risk: BANKRUPTCY\nEvidence: Current Ratio=0.1", "BANKRUPTCY", 1.0),
    ("Risk: HEALTHY\nEvidence: Current Ratio=0.1", "BANKRUPTCY", 0.0),
    ("unknown", "HEALTHY", 0.0),
]

_LABEL_RE = re.compile(r"(?im)^\s*Risk\s*:\s*(BANKRUPTCY|HEALTHY)\s*$")


def _expected_label(ground_truth: Any) -> str:
    text = str(ground_truth or "").strip().upper()
    if text in {"0", "HEALTHY"}:
        return "HEALTHY"
    if text in {"1", "BANKRUPTCY"}:
        return "BANKRUPTCY"
    return text


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    text = str(solution_str or "")
    match = _LABEL_RE.search(text)
    predicted = match.group(1).upper() if match else None
    expected = _expected_label(ground_truth)
    label_valid = predicted in {"BANKRUPTCY", "HEALTHY"}
    correct = bool(label_valid and predicted == expected)

    return {
        "score": 1.0 if correct else 0.0,
        "acc": correct,
        "pred": predicted or "[INVALID]",
        "expected": expected,
        "label_valid": label_valid,
        "outcome_correct": correct,
        "format_valid": bool(match),
        "reward_rule_id": "credit_risk_outcome_v1",
        "reward_components": {
            "outcome": {
                "raw": 1.0 if correct else 0.0,
                "weight": 1.0,
                "weighted": 1.0 if correct else 0.0,
            }
        },
    }
