import json

import pytest

from tools.amlsim.amlsim_reward import compute_score, evidence_metrics, parse_output


TRUTH = json.dumps({"risk": "SUSPICIOUS", "typology": "FAN_IN", "evidence": ["TX000001_00", "TX000001_01"]})


def test_perfect_suspicious_output_receives_maximum_reward():
    result = compute_score(None, "Risk: SUSPICIOUS\nTypology: FAN_IN\nEvidence: TX000001_00, TX000001_01", TRUTH)
    assert result["score"] == 4.0
    assert result["acc"] is True


def test_partial_evidence_receives_dense_intermediate_reward():
    result = compute_score(None, "Risk: SUSPICIOUS\nTypology: FAN_IN\nEvidence: TX000001_00", TRUTH)
    assert result["score"] == pytest.approx(10 / 3)
    assert result["evidence_f1"] == pytest.approx(2 / 3)


def test_wrong_decision_receives_minimum_reward():
    result = compute_score(None, "Risk: NORMAL\nTypology: NORMAL\nEvidence: NONE", TRUTH)
    assert result["score"] == -4.0


def test_exact_three_line_contract_rejects_extra_explanation():
    response = "Risk: SUSPICIOUS\nTypology: FAN_IN\nEvidence: TX000001_00, TX000001_01\nExplanation: extra"
    assert parse_output(response)["format_valid"] is False
    assert compute_score(None, response, TRUTH)["score"] == -4.0


def test_empty_normal_evidence_is_perfect_match():
    assert evidence_metrics([], []) == (1.0, 1.0, 1.0)

def test_empty_qwen_think_wrapper_is_ignored():
    response = "<think>\n\n</think>\n\nRisk: SUSPICIOUS\nTypology: FAN_IN\nEvidence: TX000001_00, TX000001_01"
    assert parse_output(response)["format_valid"] is True
    assert compute_score(None, response, TRUTH)["score"] == 4.0

def test_duplicate_evidence_ids_are_penalized():
    response = "Risk: SUSPICIOUS\nTypology: FAN_IN\nEvidence: TX000001_00, TX000001_01, TX000001_00"
    result = compute_score(None, response, TRUTH)
    assert result["score"] == pytest.approx(3.6)
    assert result["evidence_precision"] == pytest.approx(2 / 3)
    assert result["acc"] is False
