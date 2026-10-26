"""Tests for REQ-02: Rollout JSONL parser and Group recovery."""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_diagnostics.parsers.rollout_parser import (
    parse_rollout_dir,
    scan_rollout_dir,
)
from rl_diagnostics.schemas.data_models import content_fingerprint

GOLDEN_ROLLOUT_DIR = os.path.join(os.path.dirname(__file__), "golden_data", "golden_rollouts")
EXP_ID = "test_exp"


class TestScanRolloutDir:
    def test_finds_valid_files(self):
        step_files, warnings = scan_rollout_dir(GOLDEN_ROLLOUT_DIR)
        steps = [s for s, _ in step_files]
        assert 1 in steps
        assert 2 in steps
        assert 3 in steps
        assert 4 in steps
        assert 5 in steps

    def test_invalid_filename_warning(self):
        step_files, warnings = scan_rollout_dir(GOLDEN_ROLLOUT_DIR)
        invalid_warnings = [w for w in warnings if w["type"] == "invalid_filename"]
        assert len(invalid_warnings) >= 1
        assert any(w["filename"] == "abc.jsonl" for w in invalid_warnings)

    def test_sorted_by_step(self):
        step_files, _ = scan_rollout_dir(GOLDEN_ROLLOUT_DIR)
        steps = [s for s, _ in step_files]
        assert steps == sorted(steps)


class TestParseRolloutDir:
    def test_basic_parsing(self):
        rollouts, groups, warnings = parse_rollout_dir(GOLDEN_ROLLOUT_DIR, EXP_ID)
        assert len(rollouts) > 0
        assert len(groups) > 0

    def test_group_recovery_step1(self):
        """Step 1 has 2 groups (A: 5 rollouts, B: 5 rollouts)."""
        rollouts, groups, warnings = parse_rollout_dir(GOLDEN_ROLLOUT_DIR, EXP_ID)
        step1_groups = [g for g in groups if g.step == 1]
        assert len(step1_groups) == 2
        
        # Find group A and B
        group_a = [g for g in step1_groups if g.group_id == "group_a_step1"]
        group_b = [g for g in step1_groups if g.group_id == "group_b_step1"]
        assert len(group_a) == 1
        assert len(group_b) == 1
        assert group_a[0].rollout_count == 5
        assert group_b[0].rollout_count == 5

    def test_reward_extraction(self):
        """Rewards should be correctly extracted."""
        rollouts, groups, _ = parse_rollout_dir(GOLDEN_ROLLOUT_DIR, EXP_ID)
        group_a = [g for g in groups if g.group_id == "group_a_step1"][0]
        # Rewards: [1.0, -1.0, 1.0, -1.0, 1.0]
        assert group_a.rewards == [1.0, -1.0, 1.0, -1.0, 1.0]

    def test_constant_score_group(self):
        """Group B has all rewards = 1.0 (constant_score pattern)."""
        rollouts, groups, _ = parse_rollout_dir(GOLDEN_ROLLOUT_DIR, EXP_ID)
        group_b = [g for g in groups if g.group_id == "group_b_step1"][0]
        assert all(r == 1.0 for r in group_b.rewards)

    def test_near_constant_group(self):
        """Group C has near-constant rewards (0.49-0.51)."""
        rollouts, groups, _ = parse_rollout_dir(GOLDEN_ROLLOUT_DIR, EXP_ID)
        group_c = [g for g in groups if g.group_id == "group_c_step2"]
        assert len(group_c) == 1
        assert len(group_c[0].rewards) == 5

    def test_missing_reward_handling(self):
        """Group D has missing/nan rewards - should be handled gracefully."""
        rollouts, groups, _ = parse_rollout_dir(GOLDEN_ROLLOUT_DIR, EXP_ID)
        group_d = [g for g in groups if g.group_id == "group_d_step2"]
        assert len(group_d) == 1
        # Only 2 valid rewards (others are missing or nan)
        valid_rewards = [r for r in group_d[0].rewards if r is not None]
        assert len(valid_rewards) == 2

    def test_array_form_parsing(self):
        """Step 3 has array-form record (single record, 5 responses)."""
        rollouts, groups, _ = parse_rollout_dir(GOLDEN_ROLLOUT_DIR, EXP_ID)
        step3_groups = [g for g in groups if g.step == 3]
        assert len(step3_groups) == 1
        assert step3_groups[0].rollout_count == 5
        assert step3_groups[0].rewards == [1.0, -1.0, 1.0, -1.0, 1.0]

    def test_json_corrupted_line(self):
        """Step 4 has a corrupted JSON line - should report error, not crash."""
        rollouts, groups, warnings = parse_rollout_dir(GOLDEN_ROLLOUT_DIR, EXP_ID)
        json_errors = [r for r in rollouts if r.parse_status == "json_error"]
        assert len(json_errors) >= 1
        json_warnings = [w for w in warnings if w["type"] == "json_decode_error"]
        assert len(json_warnings) >= 1

    def test_schema_error_mismatch(self):
        """Step 4 has response/reward array length mismatch."""
        rollouts, groups, _ = parse_rollout_dir(GOLDEN_ROLLOUT_DIR, EXP_ID)
        schema_errors = [r for r in rollouts if r.parse_status == "schema_error"]
        assert len(schema_errors) >= 1

    def test_no_prompt_no_id(self):
        """Step 5 has no prompt and no ID - should still parse."""
        rollouts, groups, _ = parse_rollout_dir(GOLDEN_ROLLOUT_DIR, EXP_ID)
        step5_rollouts = [r for r in rollouts if r.step == 5]
        assert len(step5_rollouts) == 2
        # Should have content_fingerprint or unknown id_type
        for r in step5_rollouts:
            assert r.prompt is None or r.prompt == "None"

    def test_step_from_filename(self):
        """Step should come from filename and be correct."""
        rollouts, groups, _ = parse_rollout_dir(GOLDEN_ROLLOUT_DIR, EXP_ID)
        steps_found = set(r.step for r in rollouts)
        assert {1, 2, 3, 4, 5}.issubset(steps_found)

    def test_ground_truth_extraction(self):
        """Ground truth should be extracted from gts field."""
        rollouts, groups, _ = parse_rollout_dir(GOLDEN_ROLLOUT_DIR, EXP_ID)
        group_a = [g for g in groups if g.group_id == "group_a_step1"][0]
        assert group_a.ground_truth == "4"

    def test_data_source_extraction(self):
        """data_source should be extracted from record."""
        rollouts, groups, _ = parse_rollout_dir(GOLDEN_ROLLOUT_DIR, EXP_ID)
        group_a = [g for g in groups if g.group_id == "group_a_step1"][0]
        assert group_a.source_dataset == "math"

    def test_verifier_evidence_extraction(self, tmp_path):
        rollout_dir = tmp_path / "rollouts"
        rollout_dir.mkdir()
        record = {
            "input": "question",
            "output": "Answer: 4",
            "score": 1.0,
            "acc": True,
            "pred": "4",
            "uid": "group-1",
        }
        (rollout_dir / "1.jsonl").write_text(json.dumps(record) + "\n")

        rollouts, _, _ = parse_rollout_dir(str(rollout_dir), EXP_ID)

        assert rollouts[0].verifier_correct is True
        assert rollouts[0].prediction == "4"

    def test_advantage_evidence_extraction(self, tmp_path):
        rollout_dir = tmp_path / "rollouts"
        rollout_dir.mkdir()
        records = [
            {"input": "question", "output": "a", "score": 1.0, "advantage": 0.75,
             "adv_estimator": "grpo", "uid": "group-1"},
            {"input": "question", "output": "b", "score": 0.0, "advantage": -0.75,
             "adv_estimator": "grpo", "uid": "group-1"},
        ]
        (rollout_dir / "1.jsonl").write_text("\n".join(json.dumps(record) for record in records))

        rollouts, groups, _ = parse_rollout_dir(str(rollout_dir), EXP_ID)

        assert [item.advantage for item in rollouts] == [0.75, -0.75]
        assert rollouts[0].advantage_estimator == "grpo"
        assert groups[0].advantages == [0.75, -0.75]
        assert groups[0].advantage_estimator == "grpo"

    def test_content_fingerprint_deterministic(self):
        """Content fingerprint should be deterministic."""
        fp1 = content_fingerprint("hello world")
        fp2 = content_fingerprint("hello world")
        assert fp1 == fp2
        assert fp1 != content_fingerprint("hello world!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
