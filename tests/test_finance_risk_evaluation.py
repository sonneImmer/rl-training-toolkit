from tools.finance_risk.evaluate_finance_risk import calculate_metrics, parse_label


def test_parse_label_accepts_expected_two_line_format():
    assert parse_label("Risk: BANKRUPTCY\nEvidence: Current Ratio=0.1") == 1
    assert parse_label("Risk: HEALTHY\nEvidence: Current Ratio=1.2") == 0
    assert parse_label("undecided") is None


def test_calculate_metrics_counts_invalid_positive_as_false_negative():
    result = calculate_metrics([1, 1, 0, 0], [1, None, 1, 0])
    assert result["true_positive"] == 1
    assert result["false_negative"] == 1
    assert result["false_positive"] == 1
    assert result["true_negative"] == 1
    assert result["invalid_output"] == 1
    assert result["recall"] == 0.5
