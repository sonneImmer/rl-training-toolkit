"""Tests for REQ-03 Layer 1: Rollout data quality assessment."""
import math
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_diagnostics.schemas.data_models import RolloutRecord, SCHEMA_VERSION
from rl_diagnostics.analyzers.rollout_quality import assess_rollout_quality


def _make_rollout(response="hello", reward=1.0, finish_reason=None,
                   max_resp_len=None, resp_tok_len=None, parse_status="ok",
                   parse_error=None, group_id="g1", rollout_id=0, step=1) -> RolloutRecord:
    return RolloutRecord(
        schema_version=SCHEMA_VERSION,
        experiment_id="test", step=step, epoch=0, generation_attempt=None,
        group_id=group_id, rollout_id=rollout_id, prompt_id="pid",
        prompt="test prompt", response=response, training_reward=reward,
        ground_truth="42", difficulty=None, source_dataset="math",
        parse_status=parse_status, parse_error=parse_error, finish_reason=finish_reason,
        response_token_length=resp_tok_len, max_response_length=max_resp_len,
    )


class TestUsable:
    def test_normal_rollout(self):
        r = _make_rollout(response="The answer is 4.", reward=1.0)
        q = assess_rollout_quality(r)
        assert q.status == "usable"

    def test_negative_reward(self):
        r = _make_rollout(response="Wrong answer.", reward=-2.0)
        q = assess_rollout_quality(r)
        assert q.status == "usable"

    def test_zero_reward(self):
        r = _make_rollout(response="Some answer.", reward=0.0)
        q = assess_rollout_quality(r)
        assert q.status == "usable"


class TestInvalidEmpty:
    def test_empty_response(self):
        r = _make_rollout(response="", reward=1.0)
        q = assess_rollout_quality(r)
        assert q.status == "invalid_empty"

    def test_whitespace_only(self):
        r = _make_rollout(response="   \n  ", reward=1.0)
        q = assess_rollout_quality(r)
        assert q.status == "invalid_empty"

    def test_none_response(self):
        r = _make_rollout(response=None, reward=1.0)
        q = assess_rollout_quality(r)
        assert q.status == "invalid_empty"


class TestInvalidReward:
    def test_missing_reward(self):
        r = _make_rollout(response="Answer.", reward=None)
        q = assess_rollout_quality(r)
        assert q.status == "invalid_reward"

    def test_nan_reward(self):
        r = _make_rollout(response="Answer.", reward=float("nan"))
        q = assess_rollout_quality(r)
        assert q.status == "invalid_reward"

    def test_inf_reward(self):
        r = _make_rollout(response="Answer.", reward=float("inf"))
        q = assess_rollout_quality(r)
        assert q.status == "invalid_reward"


class TestInvalidTruncated:
    def test_finish_reason_length(self):
        r = _make_rollout(response="Long text...", reward=1.0, finish_reason="length")
        q = assess_rollout_quality(r)
        assert q.status == "invalid_truncated"

    def test_token_length_at_max(self):
        r = _make_rollout(response="Text", reward=1.0, max_resp_len=1024, resp_tok_len=1024)
        q = assess_rollout_quality(r)
        assert q.status == "invalid_truncated"

    def test_not_truncated(self):
        r = _make_rollout(response="Text", reward=1.0, finish_reason="stop", max_resp_len=1024, resp_tok_len=100)
        q = assess_rollout_quality(r)
        assert q.status == "usable"


class TestParseErrors:
    def test_json_error(self):
        r = _make_rollout(parse_status="json_error", parse_error="bad json")
        q = assess_rollout_quality(r)
        assert q.status == "invalid_reward"

    def test_schema_error(self):
        r = _make_rollout(parse_status="schema_error", parse_error="length mismatch")
        q = assess_rollout_quality(r)
        assert q.status == "invalid_reward"


class TestFormatChecker:
    def test_format_pass(self):
        def checker(text):
            return True, ""
        r = _make_rollout(response="Answer: 42", reward=1.0)
        q = assess_rollout_quality(r, format_checker=checker)
        assert q.status == "usable"

    def test_format_fail(self):
        def checker(text):
            return False, "No boxed answer found"
        r = _make_rollout(response="I think it's 42", reward=1.0)
        q = assess_rollout_quality(r, format_checker=checker)
        assert q.status == "invalid_format"
        assert "boxed" in q.reason

    def test_format_checker_exception(self):
        def checker(text):
            raise RuntimeError("checker crashed")
        r = _make_rollout(response="Answer", reward=1.0)
        q = assess_rollout_quality(r, format_checker=checker)
        assert q.status == "invalid_format"
        assert "crashed" in q.reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
