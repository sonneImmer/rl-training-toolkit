"""Deterministic reward for binary company bankruptcy-risk classification."""
from __future__ import annotations

import re


_LABEL_RE = re.compile(r"(?im)^\s*Risk\s*:\s*(BANKRUPTCY|HEALTHY)\s*$")
_EVIDENCE_RE = re.compile(r"(?im)^\s*Evidence\s*:\s*(.+)$")


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    expected = str(ground_truth).strip().upper()
    text = str(solution_str or "").strip()
    label_match = _LABEL_RE.search(text)
    evidence_match = _EVIDENCE_RE.search(text)
    predicted = label_match.group(1) if label_match else None
    feature_names = (extra_info or {}).get("feature_names", [])
    evidence = evidence_match.group(1) if evidence_match else ""
    cited = [name for name in feature_names if str(name).lower() in evidence.lower()]
    format_ok = label_match is not None and evidence_match is not None
    evidence_ok = len(cited) >= 2
    correct = predicted == expected

    if correct and format_ok and evidence_ok:
        score = 4.0
    elif correct:
        score = 2.0
    elif predicted is not None:
        score = -2.0
    else:
        score = -4.0

    return {
        "score": score,
        "acc": bool(correct),
        "pred": predicted or "[INVALID]",
        "format_ok": bool(format_ok),
        "evidence_ok": bool(evidence_ok),
        "evidence_count": len(cited),
    }
