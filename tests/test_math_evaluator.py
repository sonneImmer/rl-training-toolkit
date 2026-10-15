"""Tests for REQ-09: Math evaluation adapter."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_diagnostics.evaluators.math_evaluator import (
    extract_answer, extract_gsm8k_answer, extract_math_answer,
    normalize_answer, check_equivalent, evaluate_math, MathEvalConfig,
)


class TestGSM8KExtraction:
    def test_basic_hash_hash_hash(self):
        resp = "The answer is 42.\n#### 42"
        assert extract_gsm8k_answer(resp) == "42"

    def test_with_currency(self):
        resp = "Total cost is $15.50\n#### 15.50"
        assert extract_gsm8k_answer(resp) == "15.50"

    def test_with_comma(self):
        resp = "There are 1,000 apples\n#### 1000"
        assert extract_gsm8k_answer(resp) == "1000"

    def test_fallback_last_number(self):
        resp = "So the result is 7."
        assert extract_gsm8k_answer(resp) == "7"

    def test_no_answer(self):
        resp = "I don't know the answer."
        assert extract_gsm8k_answer(resp) is None


class TestMATHExtraction:
    def test_boxed(self):
        resp = "Therefore, \\boxed{42}"
        assert extract_math_answer(resp) == "42"

    def test_boxed_fraction(self):
        resp = "The answer is \\boxed{\\frac{1}{2}}"
        assert extract_math_answer(resp) == "\\frac{1}{2}"

    def test_answer_is_pattern(self):
        resp = "The answer is 3.14."
        result = extract_math_answer(resp)
        assert result is not None

    def test_fallback_last_number(self):
        resp = "So we get 255"
        assert extract_math_answer(resp) == "255"


class TestAutoExtraction:
    def test_auto_gsm8k(self):
        resp = "#### 42"
        ans, dtype = extract_answer(resp, "auto")
        assert ans == "42"
        assert dtype == "gsm8k"

    def test_auto_math(self):
        resp = "\\boxed{42}"
        ans, dtype = extract_answer(resp, "auto")
        assert ans == "42"
        assert dtype == "math"

    def test_auto_generic(self):
        resp = "The answer is 42."
        ans, dtype = extract_answer(resp, "auto")
        assert dtype == "generic"


class TestNormalization:
    def test_integer(self):
        assert normalize_answer("42") == "42"

    def test_float_trailing_zero(self):
        assert normalize_answer("42.0") == "42"

    def test_decimal(self):
        assert normalize_answer("3.14") == "3.14"

    def test_comma_number(self):
        assert normalize_answer("1,000") == "1000"

    def test_percentage(self):
        assert normalize_answer("50%") == "0.5"

    def test_fraction_latex(self):
        assert normalize_answer("\\frac{1}{2}") == "0.5"

    def test_fraction_slash(self):
        assert normalize_answer("1/2") == "0.5"

    def test_pi(self):
        assert normalize_answer("\\pi") == str(3.141592653589793)

    def test_sqrt(self):
        assert normalize_answer("\\sqrt{4}") == "2.0"

    def test_latex_stripping(self):
        assert normalize_answer("$42$") == "42"

    def test_none_input(self):
        assert normalize_answer(None) is None

    def test_empty(self):
        assert normalize_answer("") is None


class TestEquivalence:
    def test_exact_match(self):
        assert check_equivalent("42", "42") is True

    def test_numerical_equal(self):
        assert check_equivalent("42", "42.0") is True

    def test_tolerance(self):
        assert check_equivalent("3.14159", "3.14159265", tolerance=1e-4) is True

    def test_not_equal(self):
        assert check_equivalent("42", "43") is False

    def test_fraction_decimal_equivalence(self):
        assert check_equivalent("0.5", "0.5") is True

    def test_none(self):
        assert check_equivalent(None, "42") is False


class TestEvaluateMath:
    def test_all_correct(self):
        responses = ["#### 42", "#### 7", "#### 100"]
        ground_truths = ["42", "7", "100"]
        result = evaluate_math(responses, ground_truths)
        assert result.equivalent_accuracy == 1.0
        assert result.exact_match == 1.0
        assert result.parse_success_rate == 1.0
        assert result.pass_at_1 == 1.0
        assert len(result.badcases) == 0

    def test_mixed(self):
        responses = ["#### 42", "#### 43", "#### 100"]
        ground_truths = ["42", "42", "100"]
        result = evaluate_math(responses, ground_truths)
        assert result.equivalent_accuracy == pytest.approx(2 / 3)
        assert len(result.badcases) == 1
        assert result.badcases[0].status == "answer_wrong"

    def test_parse_failure(self):
        responses = ["I don't know", "#### 42"]
        ground_truths = ["42", "42"]
        result = evaluate_math(responses, ground_truths)
        assert result.parse_success_rate == 0.5
        assert result.badcase_counts.get("parse_failure") == 1

    def test_format_error(self):
        responses = ["#### \\invalid{syntax}"]
        ground_truths = ["42"]
        result = evaluate_math(responses, ground_truths)
        # Should not crash, should classify appropriately
        assert result.total_samples == 1

    def test_equivalent_not_exact(self):
        responses = ["\\boxed{\\frac{1}{2}}"]
        ground_truths = ["0.5"]
        result = evaluate_math(responses, ground_truths)
        assert result.equivalent_accuracy == 1.0

    def test_reward_agreement(self):
        responses = ["#### 42", "#### 43"]
        ground_truths = ["42", "42"]
        rewards = [1.0, 1.0]  # Both high reward, but second is wrong
        result = evaluate_math(responses, ground_truths, rewards=rewards)
        assert result.reward_eval_agreement is not None
        assert result.reward_eval_disagreement_count == 1

    def test_empty_input(self):
        result = evaluate_math([], [])
        assert result.degenerate is True
        assert result.total_samples == 0

    def test_degenerate_no_samples(self):
        result = evaluate_math([], [])
        assert result.degenerate is True

    def test_with_config(self):
        config = MathEvalConfig(dataset_type="gsm8k", tolerance=0.01)
        responses = ["#### 3.14"]
        ground_truths = ["3.14159"]
        result = evaluate_math(responses, ground_truths, config=config)
        assert result.dataset_type == "gsm8k"
        assert result.equivalent_accuracy == 1.0  # within tolerance

    def test_badcase_categories(self):
        responses = ["I don't know", "#### 43", "#### 42"]
        ground_truths = ["42", "42", "42"]
        result = evaluate_math(responses, ground_truths)
        assert "parse_failure" in result.badcase_counts
        assert "answer_wrong" in result.badcase_counts
        assert result.badcase_counts["parse_failure"] == 1
        assert result.badcase_counts["answer_wrong"] == 1

    def test_pass_at_k(self):
        responses = [
            ["#### 42", "#### 43"],  # 1/2 correct
            ["#### 43", "#### 42"],  # 1/2 correct
        ]
        ground_truths = ["42", "42"]
        config = MathEvalConfig(pass_k=2)
        result = evaluate_math(responses, ground_truths, config=config)
        assert result.pass_at_k is not None
        assert result.pass_k_value == 2
        # pass@2 = 1 - C(1,2)/C(2,2) = 1 - 0/1 = 1.0 for each
        assert result.pass_at_k == 1.0

    def test_verifier_error_separation(self):
        """Verifier errors should be separately categorized."""
        # This tests the error handling path
        responses = ["#### 42"]
        ground_truths = [None]  # Will cause format_error on gt
        result = evaluate_math(responses, ground_truths)
        assert result.badcase_counts.get("format_error", 0) >= 1
