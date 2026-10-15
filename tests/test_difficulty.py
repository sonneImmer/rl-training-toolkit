"""Tests for REQ-06: Difficulty integration."""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_diagnostics.schemas.data_models import GroupRecord, SCHEMA_VERSION
from rl_diagnostics.analyzers.difficulty_analyzer import (
    assign_difficulty, stratify_by_difficulty, load_difficulty_map,
)


def _make_group(gid="g1", difficulty=None, prompt_id="pid1", rewards=None) -> GroupRecord:
    if rewards is None:
        rewards = [1.0, -1.0, 1.0, -1.0, 1.0]
    return GroupRecord(
        schema_version=SCHEMA_VERSION, experiment_id="exp1", step=1, epoch=0,
        generation_attempt=None, group_id=gid, prompt_id=prompt_id,
        prompt="test", source_dataset="math", difficulty=difficulty,
        ground_truth="42", rollout_count=len(rewards), rewards=rewards,
        responses=[f"r{i}" for i in range(len(rewards))],
        rollout_ids=list(range(len(rewards))),
    )


class TestAssignDifficulty:
    def test_from_dataset_field(self):
        groups = [_make_group(difficulty="easy"), _make_group(difficulty="hard")]
        updated, report = assign_difficulty(groups)
        assert updated[0].difficulty == "easy"
        assert updated[1].difficulty == "hard"
        assert report.has_difficulty is True
        assert report.missing_rate == 0.0

    def test_from_external_map(self):
        groups = [_make_group(prompt_id="pid1"), _make_group(prompt_id="pid2")]
        ext_map = {"pid1": "easy", "pid2": "hard"}
        updated, report = assign_difficulty(groups, external_map=ext_map)
        assert updated[0].difficulty == "easy"
        assert updated[1].difficulty == "hard"

    def test_missing_defaults_unknown(self):
        groups = [_make_group(prompt_id="pid_unknown")]
        updated, report = assign_difficulty(groups)
        assert updated[0].difficulty == "unknown"
        assert report.missing_rate == 1.0

    def test_all_missing_skips_analysis(self):
        groups = [_make_group(), _make_group()]
        updated, report = assign_difficulty(groups)
        assert report.has_difficulty is False
        assert report.missing_rate == 1.0


class TestArbitraryLabels:
    def test_numeric_label(self):
        groups = [_make_group(difficulty="3")]
        updated, _ = assign_difficulty(groups)
        assert updated[0].difficulty == "3"

    def test_custom_string_label(self):
        groups = [_make_group(difficulty="level_alpha")]
        updated, _ = assign_difficulty(groups)
        assert updated[0].difficulty == "level_alpha"


class TestNoDifficulty:
    def test_no_error_when_absent(self):
        groups = [_make_group(), _make_group(), _make_group()]
        updated, report = assign_difficulty(groups)
        # Should not raise, all set to "unknown"
        assert all(g.difficulty == "unknown" for g in updated)
        assert report.has_difficulty is False


class TestLoadMap:
    def test_load_dict_format(self):
        path = "/tmp/test_diff_map.json"
        with open(path, "w") as f:
            json.dump({"pid1": "easy", "pid2": "hard"}, f)
        mapping, conflicts = load_difficulty_map(path)
        assert mapping["pid1"] == "easy"
        assert mapping["pid2"] == "hard"
        assert len(conflicts) == 0
        os.unlink(path)

    def test_load_list_format(self):
        path = "/tmp/test_diff_map2.json"
        with open(path, "w") as f:
            json.dump([{"prompt_id": "pid1", "difficulty": "easy"},
                       {"prompt_id": "pid2", "difficulty": "hard"}], f)
        mapping, conflicts = load_difficulty_map(path)
        assert mapping["pid1"] == "easy"
        assert mapping["pid2"] == "hard"
        os.unlink(path)


class TestStratification:
    def test_stratify_by_difficulty(self):
        from rl_diagnostics.analyzers.group_signal import GroupSignalResult
        groups = [
            _make_group(gid="g1", difficulty="easy", rewards=[1.0, -1.0, 1.0]),
            _make_group(gid="g2", difficulty="hard", rewards=[1.0, 1.0, 1.0]),
            _make_group(gid="g3", difficulty="easy", rewards=[0.5, 0.0, 1.0]),
        ]
        groups, _ = assign_difficulty(groups)
        signals = [
            GroupSignalResult(experiment_id="exp1", step=1, group_id="g1",
                              signal_type="effective_group", reason="ok",
                              rollout_count=3, usable_count=3, informative_pair_ratio=0.8,
                              rewards=[1.0, -1.0, 1.0], reward_range=2.0, reward_std=0.94),
            GroupSignalResult(experiment_id="exp1", step=1, group_id="g2",
                              signal_type="constant_score", reason="same",
                              rollout_count=3, usable_count=3, informative_pair_ratio=0.0,
                              rewards=[1.0, 1.0, 1.0], reward_range=0.0, reward_std=0.0),
            GroupSignalResult(experiment_id="exp1", step=1, group_id="g3",
                              signal_type="effective_group", reason="ok",
                              rollout_count=3, usable_count=3, informative_pair_ratio=0.7,
                              rewards=[0.5, 0.0, 1.0], reward_range=1.0, reward_std=0.47),
        ]
        buckets = stratify_by_difficulty(groups, signals)
        assert "easy" in buckets
        assert "hard" in buckets
        assert buckets["easy"].group_count == 2
        assert buckets["hard"].group_count == 1
        assert buckets["easy"].effective_group_ratio == 1.0  # both easy are effective
        assert buckets["hard"].effective_group_ratio == 0.0  # hard is constant


class TestNoInferringFromReward:
    def test_does_not_create_difficulty_from_reward(self):
        """Difficulty must not be inferred from reward values."""
        groups = [_make_group(rewards=[1.0, 1.0, 1.0])]  # all correct reward
        updated, report = assign_difficulty(groups)
        # Must be "unknown", not "easy" or "correct"
        assert updated[0].difficulty == "unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
