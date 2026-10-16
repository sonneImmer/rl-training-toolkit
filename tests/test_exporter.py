"""Tests for REQ-05: Abnormal prompt exporter."""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_diagnostics.schemas.data_models import GroupRecord, RolloutRecord, SCHEMA_VERSION
from rl_diagnostics.analyzers.rollout_quality import RolloutQualityResult
from rl_diagnostics.analyzers.group_signal import GroupSignalResult
from rl_diagnostics.exporters.prompt_exporter import export_prompts, ExportFilter


def _make_group(gid, step, rewards, prompt="test prompt", difficulty=None,
                 group_status="resolved") -> GroupRecord:
    return GroupRecord(
        schema_version=SCHEMA_VERSION, experiment_id="exp1", step=step, epoch=0,
        generation_attempt=None, group_id=gid, prompt_id=f"pid_{gid}",
        prompt=prompt, source_dataset="math", difficulty=difficulty,
        ground_truth="42", rollout_count=len(rewards), rewards=rewards,
        responses=[f"resp{i}" for i in range(len(rewards))],
        rollout_ids=list(range(len(rewards))), group_status=group_status,
    )


def _make_rollouts(gid, step, rewards, prompt="test prompt") -> list:
    return [RolloutRecord(
        schema_version=SCHEMA_VERSION, experiment_id="exp1", step=step, epoch=0,
        generation_attempt=None, group_id=gid, rollout_id=i,
        prompt_id=f"pid_{gid}", prompt=prompt, response=f"resp{i}",
        training_reward=r, ground_truth="42", source_dataset="math",
    ) for i, r in enumerate(rewards)]


class TestBasicExport:
    def test_exports_constant_score_group(self):
        groups = [_make_group("g1", 1, [1.0, 1.0, 1.0, 1.0, 1.0])]
        rollouts = _make_rollouts("g1", 1, [1.0, 1.0, 1.0, 1.0, 1.0])
        quality = [RolloutQualityResult(
            experiment_id="exp1", step=1, group_id="g1", rollout_id=i, status="usable"
        ) for i in range(5)]
        signals = [GroupSignalResult(
            experiment_id="exp1", step=1, group_id="g1",
            signal_type="constant_score", reason="all same",
            rollout_count=5, usable_count=5,
        )]

        path = "/tmp/test_export_basic.jsonl"
        count, blocked, blocked_recs = export_prompts(
            groups, rollouts, quality, signals, path
        )
        assert count == 1
        assert blocked == 0

        with open(path) as f:
            entry = json.loads(f.readline())
        assert entry["diagnostic_type"] == "constant_score"
        assert entry["prompt"] == "test prompt"
        assert "manual_review" in entry["suggested_usage"]
        assert "resampling" in entry["suggested_usage"]
        os.unlink(path)

    def test_skips_effective_groups(self):
        groups = [
            _make_group("g1", 1, [1.0, -1.0, 1.0, -1.0, 1.0]),  # effective
            _make_group("g2", 1, [1.0, 1.0, 1.0]),  # constant
        ]
        rollouts = _make_rollouts("g1", 1, [1.0, -1.0, 1.0, -1.0, 1.0]) + \
                   _make_rollouts("g2", 1, [1.0, 1.0, 1.0])
        quality = [RolloutQualityResult(
            experiment_id="exp1", step=1, group_id="g1", rollout_id=i, status="usable"
        ) for i in range(5)] + [RolloutQualityResult(
            experiment_id="exp1", step=1, group_id="g2", rollout_id=i, status="usable"
        ) for i in range(3)]
        signals = [
            GroupSignalResult(experiment_id="exp1", step=1, group_id="g1",
                              signal_type="effective_group", reason="ok", rollout_count=5, usable_count=5),
            GroupSignalResult(experiment_id="exp1", step=1, group_id="g2",
                              signal_type="constant_score", reason="all same", rollout_count=3, usable_count=3),
        ]

        path = "/tmp/test_export_skip.jsonl"
        count, _, _ = export_prompts(groups, rollouts, quality, signals, path)
        assert count == 1  # only g2
        os.unlink(path)

    def test_exports_mostly_wrong_with_evidence(self):
        groups = [_make_group("g1", 1, [-1.0, -1.0, -1.0, 1.0])]
        rollouts = _make_rollouts("g1", 1, [-1.0, -1.0, -1.0, 1.0])
        for rollout, correct in zip(rollouts, [False, False, False, True]):
            rollout.verifier_correct = correct
            rollout.prediction = "candidate"
        quality = [RolloutQualityResult(
            experiment_id="exp1", step=1, group_id="g1", rollout_id=i, status="usable"
        ) for i in range(4)]
        signals = [GroupSignalResult(
            experiment_id="exp1", step=1, group_id="g1",
            signal_type="effective_group", reason="mixed reward",
            rollout_count=4, usable_count=4,
            positive_reward_ratio=0.25,
            reward_outcome_type="mostly_non_positive",
            correctness_ratio=0.25,
            correctness_type="mostly_wrong",
        )]

        path = "/tmp/test_export_mostly_wrong.jsonl"
        count, _, _ = export_prompts(groups, rollouts, quality, signals, path)
        assert count == 1
        with open(path) as f:
            entry = json.loads(f.readline())
        assert entry["diagnostic_type"] == "mostly_wrong"
        assert entry["correctness_ratio"] == pytest.approx(0.25)
        assert entry["verifier_correct"] == [False, False, False, True]
        os.unlink(path)


class TestDeduplication:
    def test_dedup_by_prompt_id(self):
        """Same prompt in multiple steps should be deduplicated but keep all triggered steps."""
        groups = [
            _make_group("g1", 1, [1.0, 1.0], prompt="same prompt"),
            _make_group("g2", 2, [1.0, 1.0], prompt="same prompt"),
        ]
        # Override prompt_ids to be the same for dedup test
        groups[0].prompt_id = "same_pid"
        groups[1].prompt_id = "same_pid"
        rollouts = _make_rollouts("g1", 1, [1.0, 1.0], "same prompt") + \
                   _make_rollouts("g2", 2, [1.0, 1.0], "same prompt")
        quality = [RolloutQualityResult(experiment_id="exp1", step=1, group_id="g1", rollout_id=0, status="usable"),
                   RolloutQualityResult(experiment_id="exp1", step=1, group_id="g1", rollout_id=1, status="usable"),
                   RolloutQualityResult(experiment_id="exp1", step=2, group_id="g2", rollout_id=0, status="usable"),
                   RolloutQualityResult(experiment_id="exp1", step=2, group_id="g2", rollout_id=1, status="usable")]
        signals = [
            GroupSignalResult(experiment_id="exp1", step=1, group_id="g1",
                              signal_type="constant_score", reason="same", rollout_count=2, usable_count=2),
            GroupSignalResult(experiment_id="exp1", step=2, group_id="g2",
                              signal_type="constant_score", reason="same", rollout_count=2, usable_count=2),
        ]

        path = "/tmp/test_export_dedup.jsonl"
        count, _, _ = export_prompts(groups, rollouts, quality, signals, path)
        assert count == 1  # deduplicated
        with open(path) as f:
            entry = json.loads(f.readline())
        assert set(entry["triggered_steps"]) == {1, 2}
        assert set(entry["group_ids"]) == {"g1", "g2"}
        os.unlink(path)


class TestBlockedExport:
    def test_blocks_unrecoverable_prompt(self):
        groups = [_make_group("g1", 1, [1.0, 1.0], prompt=None)]
        rollouts = _make_rollouts("g1", 1, [1.0, 1.0], prompt=None)
        quality = [RolloutQualityResult(experiment_id="exp1", step=1, group_id="g1", rollout_id=i, status="usable") for i in range(2)]
        signals = [GroupSignalResult(experiment_id="exp1", step=1, group_id="g1",
                                      signal_type="constant_score", reason="same", rollout_count=2, usable_count=2)]

        path = "/tmp/test_export_blocked.jsonl"
        count, blocked, blocked_recs = export_prompts(groups, rollouts, quality, signals, path)
        assert count == 0
        assert blocked == 1
        assert blocked_recs[0]["reason"] == "Prompt text not recoverable"
        os.unlink(path)


class TestExportFilter:
    def test_filter_by_step(self):
        groups = [
            _make_group("g1", 1, [1.0, 1.0]),
            _make_group("g2", 2, [1.0, 1.0]),
        ]
        rollouts = _make_rollouts("g1", 1, [1.0, 1.0]) + _make_rollouts("g2", 2, [1.0, 1.0])
        quality = [RolloutQualityResult(experiment_id="exp1", step=s, group_id=gid, rollout_id=i, status="usable")
                   for gid, s, n in [("g1", 1, 2), ("g2", 2, 2)] for i in range(n)]
        signals = [
            GroupSignalResult(experiment_id="exp1", step=1, group_id="g1", signal_type="constant_score", reason="same", rollout_count=2, usable_count=2),
            GroupSignalResult(experiment_id="exp1", step=2, group_id="g2", signal_type="constant_score", reason="same", rollout_count=2, usable_count=2),
        ]

        path = "/tmp/test_export_filter.jsonl"
        filt = ExportFilter(steps=[2])
        count, _, _ = export_prompts(groups, rollouts, quality, signals, path, export_filter=filt)
        assert count == 1
        with open(path) as f:
            entry = json.loads(f.readline())
        assert entry["step"] == 2
        os.unlink(path)

    def test_filter_by_diagnosis_type(self):
        groups = [
            _make_group("g1", 1, [1.0, 1.0]),  # constant
            _make_group("g2", 1, [1.0, 0.0, 1.0, 0.0]),  # effective (skipped)
        ]
        rollouts = _make_rollouts("g1", 1, [1.0, 1.0]) + _make_rollouts("g2", 1, [1.0, 0.0, 1.0, 0.0])
        quality = [RolloutQualityResult(experiment_id="exp1", step=1, group_id="g1", rollout_id=i, status="usable") for i in range(2)] + \
                  [RolloutQualityResult(experiment_id="exp1", step=1, group_id="g2", rollout_id=i, status="usable") for i in range(4)]
        signals = [
            GroupSignalResult(experiment_id="exp1", step=1, group_id="g1", signal_type="constant_score", reason="same", rollout_count=2, usable_count=2),
            GroupSignalResult(experiment_id="exp1", step=1, group_id="g2", signal_type="effective_group", reason="ok", rollout_count=4, usable_count=4),
        ]

        path = "/tmp/test_export_filter2.jsonl"
        filt = ExportFilter(diagnosis_types=["constant_score"])
        count, _, _ = export_prompts(groups, rollouts, quality, signals, path, export_filter=filt)
        assert count == 1
        os.unlink(path)


class TestJsonlFormat:
    def test_each_line_parseable(self):
        groups = [_make_group("g1", 1, [1.0, 1.0, 1.0])]
        rollouts = _make_rollouts("g1", 1, [1.0, 1.0, 1.0])
        quality = [RolloutQualityResult(experiment_id="exp1", step=1, group_id="g1", rollout_id=i, status="usable") for i in range(3)]
        signals = [GroupSignalResult(experiment_id="exp1", step=1, group_id="g1",
                                      signal_type="constant_score", reason="same", rollout_count=3, usable_count=3)]

        path = "/tmp/test_export_jsonl.jsonl"
        count, _, _ = export_prompts(groups, rollouts, quality, signals, path)
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                assert "experiment_id" in entry
                assert "prompt" in entry
                assert "diagnostic_type" in entry
                assert "suggested_usage" in entry
                assert "exported_at" in entry
        os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
