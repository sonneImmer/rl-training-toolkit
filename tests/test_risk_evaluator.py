"""Tests for REQ-08: Risk control evaluation adapter."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_diagnostics.evaluators.risk_evaluator import (
    evaluate_risk, RiskEvalConfig, _confusion_counts, _precision,
    _recall, _fbeta, _compute_pr_curve, _pr_auc_from_curve,
    _compute_roc_auc, _compute_ks, _brier_score,
)


class TestConfusionCounts:
    def test_basic(self):
        tp, fp, tn, fn = _confusion_counts([1, 0, 1, 0], [1, 0, 0, 1])
        assert tp == 1
        assert fp == 1
        assert tn == 1
        assert fn == 1

    def test_all_positive(self):
        tp, fp, tn, fn = _confusion_counts([1, 1, 1], [1, 1, 1])
        assert tp == 3
        assert fp == 0
        assert tn == 0
        assert fn == 0

    def test_all_wrong(self):
        tp, fp, tn, fn = _confusion_counts([1, 1], [0, 0])
        assert tp == 0
        assert fp == 0
        assert tn == 0
        assert fn == 2


class TestPrecisionRecall:
    def test_precision(self):
        assert _precision(3, 1) == 0.75

    def test_precision_zero_division(self):
        assert _precision(0, 0) == 0.0

    def test_recall(self):
        assert _recall(3, 1) == 0.75

    def test_recall_zero_division(self):
        assert _recall(0, 0) == 0.0


class TestFBeta:
    def test_f1(self):
        p, r = 0.5, 0.5
        assert _fbeta(p, r, 1.0) == 0.5

    def test_f1_zero(self):
        assert _fbeta(0, 0, 1.0) == 0.0

    def test_f2_weights_recall(self):
        p, r = 0.5, 0.8
        f1 = _fbeta(p, r, 1.0)
        f2 = _fbeta(p, r, 2.0)
        assert f2 > f1  # F2 gives more weight to recall


class TestPRCurve:
    def test_perfect_classifier(self):
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        curve = _compute_pr_curve(labels, scores)
        assert len(curve) > 0
        # First point should have high precision
        assert curve[0]["precision"] == 1.0

    def test_no_positives(self):
        labels = [0, 0, 0]
        scores = [0.9, 0.5, 0.1]
        curve = _compute_pr_curve(labels, scores)
        assert curve == []

    def test_pr_auc_perfect(self):
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        curve = _compute_pr_curve(labels, scores)
        auc = _pr_auc_from_curve(curve)
        assert auc >= 0.5

    def test_pr_auc_random(self):
        labels = [1, 0, 1, 0]
        scores = [0.4, 0.6, 0.3, 0.7]
        curve = _compute_pr_curve(labels, scores)
        auc = _pr_auc_from_curve(curve)
        # Random classifier should have AUC around positive rate
        assert 0.0 <= auc <= 1.0


class TestROCAUC:
    def test_perfect(self):
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        auc = _compute_roc_auc(labels, scores)
        assert auc == 1.0

    def test_all_same_class(self):
        assert _compute_roc_auc([1, 1], [0.5, 0.5]) == 0.0


class TestKS:
    def test_perfect_separation(self):
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        ks = _compute_ks(labels, scores)
        assert ks == 1.0

    def test_no_separation(self):
        labels = [1, 0]
        scores = [0.5, 0.5]
        ks = _compute_ks(labels, scores)
        assert ks == 0.0


class TestBrierScore:
    def test_perfect(self):
        labels = [1, 0]
        scores = [1.0, 0.0]
        assert _brier_score(labels, scores) == 0.0

    def test_worst(self):
        labels = [1, 0]
        scores = [0.0, 1.0]
        assert _brier_score(labels, scores) == 1.0


class TestEvaluateRisk:
    def test_basic_evaluation(self):
        labels = [1, 0, 1, 0, 1, 0]
        predictions = [1, 0, 1, 1, 0, 0]
        result = evaluate_risk(labels, predictions)
        assert result.total_samples == 6
        assert result.positive_count == 3
        assert result.negative_count == 3
        assert 0 <= result.precision <= 1
        assert 0 <= result.recall <= 1

    def test_degenerate_no_positives(self):
        labels = [0, 0, 0]
        predictions = [0, 0, 0]
        result = evaluate_risk(labels, predictions)
        assert result.degenerate is True
        assert "No positive" in result.degenerate_reason

    def test_degenerate_no_negatives(self):
        labels = [1, 1, 1]
        predictions = [1, 1, 1]
        result = evaluate_risk(labels, predictions)
        assert result.degenerate is True

    def test_with_scores(self):
        labels = [1, 0, 1, 0]
        predictions = [1, 0, 1, 0]
        scores = [0.9, 0.1, 0.8, 0.2]
        result = evaluate_risk(labels, predictions, scores=scores)
        assert result.roc_auc == 1.0
        assert result.pr_auc >= 0.5

    def test_with_probability(self):
        labels = [1, 0]
        predictions = [0.9, 0.1]
        config = RiskEvalConfig(has_probability=True)
        result = evaluate_risk(labels, predictions, config=config)
        assert result.brier_score is not None
        assert result.calibration_error is not None

    def test_reward_agreement(self):
        labels = [1, 0, 1, 0]
        predictions = [1, 0, 1, 0]
        rewards = [1.0, 0.0, 1.0, 0.0]
        result = evaluate_risk(labels, predictions, rewards=rewards)
        assert result.reward_eval_agreement == 1.0
        assert result.reward_eval_disagreement_count == 0

    def test_reward_disagreement(self):
        labels = [1, 0, 1, 0]
        predictions = [1, 0, 1, 0]
        rewards = [0.0, 1.0, 1.0, 0.0]  # First two disagree
        result = evaluate_risk(labels, predictions, rewards=rewards)
        assert result.reward_eval_disagreement_count == 2

    def test_top_k(self):
        labels = [1, 1, 0, 0, 0, 0]
        predictions = [1, 0, 0, 0, 0, 0]
        scores = [0.9, 0.1, 0.8, 0.7, 0.6, 0.5]
        config = RiskEvalConfig(has_probability=True, top_k=2)
        result = evaluate_risk(labels, predictions, scores=scores, config=config)
        assert result.topk_recall is not None
        # Top 2 by score: index 0 (label=1) and index 2 (label=0)
        assert result.topk_recall == 0.5  # 1 out of 2 positives

    def test_false_negative_badcases(self):
        labels = [1, 1, 0]
        predictions = [0, 1, 0]
        result = evaluate_risk(labels, predictions)
        assert result.false_negative_count == 1
        assert len(result.false_negative_badcases) == 1
        assert result.false_negative_badcases[0]["index"] == 0

    def test_empty_input(self):
        result = evaluate_risk([], [])
        assert result.total_samples == 0
        assert result.degenerate is True

    def test_positive_label_custom(self):
        labels = ["yes", "no", "yes"]
        predictions = ["yes", "yes", "yes"]
        config = RiskEvalConfig(positive_label="yes")
        result = evaluate_risk(labels, predictions, config=config)
        assert result.positive_count == 2

    def test_precision_floor(self):
        labels = [1, 1, 0, 0]
        predictions = [1, 0, 1, 0]
        scores = [0.9, 0.8, 0.2, 0.1]
        config = RiskEvalConfig(precision_floor=0.5)
        result = evaluate_risk(labels, predictions, scores=scores, config=config)
        assert result.recall_at_precision is not None
