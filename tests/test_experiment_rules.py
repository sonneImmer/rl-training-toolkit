from rl_diagnostics.experiment_rules import RuleConfig, evaluate_experiment_rules


def _row(step, **overrides):
    row = {
        "step": step,
        "rollouts": 8,
        "valid_ratio": 1.0,
        "clip_ratio": 0.0,
        "grad_norm": 1.0,
        "kl_loss": 0.01,
        "has_correctness": True,
        "all_correct_groups": 1,
        "mostly_correct_groups": 0,
        "mixed_groups": 1,
        "mostly_wrong_groups": 0,
        "all_wrong_groups": 0,
        "analyzable_groups": 4,
        "constant_score_groups": 0,
        "near_constant_groups": 0,
        "effective_groups": 4,
        "low_information_group_ratio": 0.0,
    }
    row.update(overrides)
    return row


def test_window_rules_require_consecutive_steps():
    rows = [_row(1, clip_ratio=0.3), _row(2, clip_ratio=0.1)]
    labels = evaluate_experiment_rules(rows, "running")
    assert "high_truncation" not in {label["name"] for label in labels}

    rows.append(_row(3, clip_ratio=0.4))
    rows.append(_row(4, clip_ratio=0.5))
    labels = evaluate_experiment_rules(rows, "running")
    label = next(item for item in labels if item["name"] == "high_truncation")
    assert label["observation"]
    assert label["decision_reason"]
    assert label["meaning"]
    assert label["boundary"]
    assert label["suggestions"]


def test_low_group_signal_uses_reward_separation_evidence():
    rows = [
        _row(1, constant_score_groups=3, near_constant_groups=1, low_information_group_ratio=1.0),
        _row(2, constant_score_groups=3, near_constant_groups=1, low_information_group_ratio=1.0),
    ]
    labels = evaluate_experiment_rules(rows, "running")
    label = next(item for item in labels if item["name"] == "low_group_signal")
    assert label["metric"] == "low_information_group_ratio"
    assert label["evidence"][0] == {"step": 1, "value": 1.0, "numerator": 4, "denominator": 4}
    assert "4/4" in label["observation"]
    assert "全对或全错" in label["boundary"]


def test_all_correct_or_all_wrong_does_not_imply_low_reward_signal():
    rows = [
        _row(1, all_correct_groups=4, mixed_groups=0),
        _row(2, all_wrong_groups=4, all_correct_groups=0, mixed_groups=0),
    ]
    labels = evaluate_experiment_rules(rows, "running")
    assert "low_group_signal" not in {label["name"] for label in labels}


def test_completed_with_errors_gets_specific_label_and_log_summary():
    labels = evaluate_experiment_rules([_row(2)], "completed_with_errors", error_summary="RuntimeError: cleanup failure")
    label = next(item for item in labels if item["name"] == "post_training_error")
    assert "cleanup failure" in label["observation"]
    assert "experiment_failed" not in {item["name"] for item in labels}


def test_custom_thresholds_are_applied():
    config = RuleConfig(window_steps=1, valid_ratio_threshold=0.95)
    labels = evaluate_experiment_rules([_row(1, valid_ratio=0.9)], "running", config)
    assert "low_valid_ratio" in {label["name"] for label in labels}
