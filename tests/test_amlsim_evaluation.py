from tools.amlsim.amlsim_reward import parse_output
from tools.amlsim.evaluate_amlsim import calculate_metrics


def record(expected_risk, expected_typology, expected_evidence, response):
    return {
        "expected": {"risk": expected_risk, "typology": expected_typology, "evidence": expected_evidence},
        "parsed": parse_output(response),
    }


def test_metrics_cover_risk_typology_evidence_and_invalid_output():
    records = [
        record("SUSPICIOUS", "FAN_IN", ["TX000001_00"], "Risk: SUSPICIOUS\nTypology: FAN_IN\nEvidence: TX000001_00"),
        record("NORMAL", "NORMAL", [], "unparseable"),
    ]
    metrics = calculate_metrics(records)
    assert metrics["risk"]["true_positive"] == 1
    assert metrics["risk"]["false_positive"] == 1
    assert metrics["risk"]["invalid_output"] == 1
    assert metrics["typology_accuracy"] == 0.5
    assert metrics["format_valid_rate"] == 0.5
