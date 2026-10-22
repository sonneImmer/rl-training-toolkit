"""Deterministic composite reward for AMLSim typology classification and evidence."""
from __future__ import annotations

from collections import Counter
import json
import re


RISK_LINE_RE = re.compile(r"^\s*Risk\s*:\s*(SUSPICIOUS|NORMAL)\s*$", re.IGNORECASE)
TYPOLOGY_LINE_RE = re.compile(r"^\s*Typology\s*:\s*(FAN_IN|FAN_OUT|CYCLE|NORMAL)\s*$", re.IGNORECASE)
EVIDENCE_LINE_RE = re.compile(r"^\s*Evidence\s*:\s*(.+?)\s*$", re.IGNORECASE)
TX_RE = re.compile(r"TX\d{6}_\d{2}")
REWARD_TEST_CASES = [
    ("Risk: SUSPICIOUS\nTypology: FAN_IN\nEvidence: TX000001_00, TX000001_01", json.dumps({"risk": "SUSPICIOUS", "typology": "FAN_IN", "evidence": ["TX000001_00", "TX000001_01"]}), 4.0),
    ("Risk: NORMAL\nTypology: NORMAL\nEvidence: NONE", json.dumps({"risk": "NORMAL", "typology": "NORMAL", "evidence": []}), 4.0),
]


def parse_output(text: str) -> dict:
    cleaned = re.sub(r"^\s*<think>\s*</think>\s*", "", str(text or ""), count=1, flags=re.IGNORECASE)
    lines = [line for line in cleaned.splitlines() if line.strip()]
    risk_match = RISK_LINE_RE.fullmatch(lines[0]) if len(lines) == 3 else None
    typology_match = TYPOLOGY_LINE_RE.fullmatch(lines[1]) if len(lines) == 3 else None
    evidence_match = EVIDENCE_LINE_RE.fullmatch(lines[2]) if len(lines) == 3 else None
    evidence = []
    if evidence_match and evidence_match.group(1).strip().upper() != "NONE":
        evidence = TX_RE.findall(evidence_match.group(1).upper())
    return {
        "risk": risk_match.group(1).upper() if risk_match else None,
        "typology": typology_match.group(1).upper() if typology_match else None,
        "evidence": evidence,
        "format_valid": bool(risk_match and typology_match and evidence_match),
    }


def evidence_metrics(predicted: list[str], expected: list[str]) -> tuple[float, float, float]:
    predicted_counts, expected_counts = Counter(predicted), Counter(expected)
    if not predicted_counts and not expected_counts:
        return 1.0, 1.0, 1.0
    matched = sum((predicted_counts & expected_counts).values())
    precision = matched / sum(predicted_counts.values()) if predicted_counts else 0.0
    recall = matched / sum(expected_counts.values()) if expected_counts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    expected = json.loads(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth)
    predicted = parse_output(str(solution_str or ""))
    risk_correct = predicted["risk"] == expected["risk"]
    typology_correct = predicted["typology"] == expected["typology"]
    evidence_precision, evidence_recall, evidence_f1 = evidence_metrics(predicted["evidence"], expected["evidence"])
    risk_score = 2.0 if risk_correct else -2.0
    typology_score = 1.0 if typology_correct else -1.0
    evidence_score = 2.0 * evidence_f1 - 1.0
    score = risk_score + typology_score + evidence_score
    if not predicted["format_valid"]:
        score = -4.0
    return {
        "score": float(score),
        "acc": bool(risk_correct and typology_correct and evidence_f1 == 1.0),
        "pred": predicted["typology"] or "[INVALID]",
        "expected": expected["typology"],
        "format_valid": predicted["format_valid"],
        "risk_correct": risk_correct,
        "typology_correct": typology_correct,
        "evidence_precision": evidence_precision,
        "evidence_recall": evidence_recall,
        "evidence_f1": evidence_f1,
        "reward_rule_id": "amlsim_risk_typology_evidence_v1",
        "reward_components": {
            "risk": {"raw": risk_correct, "weight": 2.0, "weighted": risk_score},
            "typology": {"raw": typology_correct, "weight": 1.0, "weighted": typology_score},
            "evidence": {"raw": evidence_f1, "weight": 1.0, "weighted": evidence_score},
        },
    }
