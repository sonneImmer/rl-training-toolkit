"""Deterministic reward for FinQA executable financial reasoning."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from finqa_program import execute_program, parse_output, values_equal
except ImportError:
    from .finqa_program import execute_program, parse_output, values_equal


def _json_value(value: Any, fallback: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return value if value is not None else fallback


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    truth = _json_value(ground_truth, {})
    metadata = _json_value(extra_info, {})
    table = _json_value(metadata.get("table"), [])
    parsed = parse_output(solution_str)
    result = execute_program(parsed["program"], table)
    gold_answer = truth.get("answer", truth.get("exe_ans"))
    execution_correct = result.valid and values_equal(result.value, gold_answer)
    answer_correct = values_equal(parsed["answer"], gold_answer)
    answer_consistent = result.valid and values_equal(parsed["answer"], result.value)

    gold_evidence = {str(item).strip().lower() for item in truth.get("gold_evidence", []) if str(item).strip()}
    predicted_evidence = {str(item).strip().lower() for item in parsed["evidence"]}
    evidence_hits = len(gold_evidence & predicted_evidence)
    evidence_recall = evidence_hits / len(gold_evidence) if gold_evidence else 1.0
    evidence_precision = evidence_hits / len(predicted_evidence) if predicted_evidence else 0.0
    evidence_f1 = (
        2 * evidence_precision * evidence_recall / (evidence_precision + evidence_recall)
        if evidence_precision + evidence_recall
        else 0.0
    )

    if not parsed["program"]:
        score = -4.0
        reason = "missing_program"
    elif not parsed["format_valid"]:
        score = -3.5
        reason = "invalid_output_format"
    elif not result.valid:
        score = -3.0
        reason = "invalid_program"
    elif not execution_correct:
        score = -2.0
        reason = "wrong_execution_result"
    elif not answer_correct or not answer_consistent:
        score = 1.0
        reason = "correct_program_inconsistent_answer"
    else:
        score = 3.5 + 0.5 * evidence_f1
        reason = "correct"

    return {
        "score": round(float(score), 6),
        "acc": bool(parsed["format_valid"] and execution_correct and answer_correct and answer_consistent),
        "pred": parsed["answer"] or "[INVALID]",
        "format_valid": bool(parsed["format_valid"]),
        "program_valid": bool(result.valid),
        "execution_correct": bool(execution_correct),
        "answer_correct": bool(answer_correct),
        "answer_consistent": bool(answer_consistent),
        "execution_result": result.value,
        "program_steps": result.steps,
        "evidence_recall": round(evidence_recall, 6),
        "evidence_precision": round(evidence_precision, 6),
        "evidence_f1": round(evidence_f1, 6),
        "evidence_hits": evidence_hits,
        "evidence_total": len(gold_evidence),
        "reward_reason": reason,
    }
