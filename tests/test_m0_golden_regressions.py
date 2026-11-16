import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rl_diagnostics.analyzers.group_signal import diagnose_group_signal
from rl_diagnostics.analyzers.rollout_quality import assess_rollout_quality
from rl_diagnostics.live_monitor import build_snapshot
from rl_diagnostics.parsers.rollout_parser import parse_rollout_dir


class M0GoldenRegressionTest(unittest.TestCase):
    def setUp(self):
        self.fixture = Path(__file__).parent / "golden_data" / "m0_rollouts.jsonl"

    def _parse_fixture(self, tmp_path):
        rollout_dir = tmp_path / "rollouts"
        rollout_dir.mkdir()
        (rollout_dir / "1.jsonl").write_text(self.fixture.read_text())
        return rollout_dir, parse_rollout_dir(str(rollout_dir), "m0-golden")

    def test_parser_marks_unrecoverable_group_identity(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            _, (_, groups, _) = self._parse_fixture(Path(directory))
            unknown = [group for group in groups if group.group_id_quality == "unknown"]
            self.assertEqual(len(unknown), 2)
            self.assertTrue(all(group.group_status == "group_unresolved" for group in unknown))

    def test_group_semantics_keep_correctness_reward_and_advantage_separate(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            _, (rollouts, groups, _) = self._parse_fixture(Path(directory))
            quality = [assess_rollout_quality(item) for item in rollouts]
            by_id = {group.group_id: group for group in groups}
            def diagnose(group_id):
                labels = [item.verifier_correct for item in rollouts if item.group_id == group_id]
                return diagnose_group_signal(by_id[group_id], quality, correctness_labels=labels)
            self.assertEqual(diagnose("all-correct").correctness_type, "all_correct")
            wrong = diagnose("all-wrong")
            self.assertEqual(wrong.correctness_type, "all_wrong")
            self.assertTrue(wrong.all_wrong_negative_advantage)
            self.assertEqual(diagnose("mixed").correctness_type, "mixed_correctness")
            self.assertEqual(diagnose("verifier-failure").correctness_type, "unknown")

    def test_live_monitor_does_not_make_unknown_singletons_into_groups(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollout_dir, _ = self._parse_fixture(root)
            snapshot = build_snapshot(root / "missing.log", rollout_dir)
            step = snapshot["steps"][0]
            self.assertEqual(step["group_id_quality"], {"explicit": 4, "unknown": 2})
            self.assertEqual(step["all_correct_groups"], 1)
            self.assertEqual(step["all_wrong_groups"], 1)

    def test_difficulty_annotation_preserves_advantage_evidence(self):
        import tempfile
        from rl_diagnostics.analyzers.difficulty_analyzer import assign_difficulty
        with tempfile.TemporaryDirectory() as directory:
            _, (_, groups, _) = self._parse_fixture(Path(directory))
            updated, _ = assign_difficulty(groups)
            wrong = next(group for group in updated if group.group_id == "all-wrong")
            self.assertEqual(wrong.advantages, [-1.0, -1.0, -1.0, -1.0])
            self.assertEqual(wrong.advantage_estimator, "ngrpo")


if __name__ == "__main__":
    unittest.main()
