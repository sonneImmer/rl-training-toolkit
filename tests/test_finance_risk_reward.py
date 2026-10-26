from tools.finance_risk.finance_risk_outcome_reward import compute_score
from tools.finance_risk.finance_risk_class_weighted_reward import compute_score as weighted_compute_score


def score(response, truth):
    return compute_score(
        data_source="finance_risk_taiwanese_bankruptcy",
        solution_str=response,
        ground_truth=truth,
        extra_info={"feature_names": ["Current Ratio", "Quick Ratio"]},
    )


def test_correct_bankruptcy_label_gets_only_outcome_reward():
    result = score("Risk: BANKRUPTCY\nEvidence: Current Ratio=0.1", "BANKRUPTCY")
    assert result["score"] == 1.0
    assert result["outcome_correct"] is True
    assert result["reward_components"]["outcome"]["weighted"] == 1.0


def test_correct_healthy_label_gets_only_outcome_reward():
    result = score("Risk: HEALTHY\nEvidence: Current Ratio=1.2", "HEALTHY")
    assert result["score"] == 1.0
    assert result["label_valid"] is True


def test_wrong_label_gets_zero_without_negative_format_penalty():
    result = score("Risk: HEALTHY\nEvidence: Current Ratio=0.1", "BANKRUPTCY")
    assert result["score"] == 0.0
    assert result["label_valid"] is True
    assert result["outcome_correct"] is False


def test_invalid_label_gets_zero_and_is_recorded():
    result = score("I cannot decide", "HEALTHY")
    assert result["score"] == 0.0
    assert result["label_valid"] is False
    assert result["pred"] == "[INVALID]"


def test_extra_text_does_not_create_an_invalid_outcome_if_label_is_parseable():
    result = score("Risk: BANKRUPTCY\nEvidence: unavailable\nExtra explanation", "BANKRUPTCY")
    assert result["score"] == 1.0
    assert result["format_valid"] is True


def test_class_weighted_reward_prioritizes_bankruptcy_errors():
    correct_positive = weighted_compute_score(None, "Risk: BANKRUPTCY", "BANKRUPTCY")
    missed_positive = weighted_compute_score(None, "Risk: HEALTHY", "BANKRUPTCY")
    correct_negative = weighted_compute_score(None, "Risk: HEALTHY", "HEALTHY")
    false_alarm = weighted_compute_score(None, "Risk: BANKRUPTCY", "HEALTHY")
    assert [correct_positive["score"], missed_positive["score"]] == [3.0, -3.0]
    assert [correct_negative["score"], false_alarm["score"]] == [1.0, -1.0]


def test_class_weighted_reward_penalizes_invalid_output():
    result = weighted_compute_score(None, "unknown", "HEALTHY")
    assert result["score"] == -3.0
    assert result["label_valid"] is False
