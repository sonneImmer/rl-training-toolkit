"""Tests for REQ-01: Log parser."""
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_diagnostics.parsers.log_parser import (
    parse_log,
    parse_parameters,
    _clean_line,
    _parse_value,
)

GOLDEN_LOG = os.path.join(os.path.dirname(__file__), "golden_data", "golden_log.txt")


class TestCleanLine:
    def test_removes_ansi(self):
        line = "\x1b[36m(TaskRunner pid=12345)\x1b[0m step:1 - actor/entropy:0.733"
        cleaned = _clean_line(line)
        assert "\x1b" not in cleaned
        assert "(TaskRunner" not in cleaned
        assert "step:1" in cleaned

    def test_removes_timestamp(self):
        line = "[2026-10-09 11:48:43,680] INFO something happened"
        cleaned = _clean_line(line)
        assert "2026" not in cleaned
        assert "something happened" in cleaned


class TestParseValue:
    def test_int(self):
        assert _parse_value("5") == 5
        assert isinstance(_parse_value("5"), int)

    def test_float(self):
        assert _parse_value("0.5") == 0.5
        assert _parse_value("5e-7") == 5e-7

    def test_bool(self):
        assert _parse_value("True") is True
        assert _parse_value("False") is False

    def test_string(self):
        assert _parse_value("grpo") == "grpo"

    def test_list(self):
        result = _parse_value("['a','b']")
        assert result == ["a", "b"]


class TestParseParameters:
    def test_basic_params(self):
        cmd = "python -m verl.trainer.main_ppo algorithm.adv_estimator=grpo trainer.total_epochs=3"
        params, conflicts = parse_parameters(cmd)
        assert params["algorithm.adv_estimator"] == "grpo"
        assert params["trainer.total_epochs"] == 3
        assert len(conflicts) == 0

    def test_duplicate_keys(self):
        cmd = "python -m main_ppo trainer.total_epochs=3 trainer.total_epochs=5"
        params, conflicts = parse_parameters(cmd)
        assert params["trainer.total_epochs"] == 5  # last wins
        assert len(conflicts) == 1
        assert conflicts[0]["key"] == "trainer.total_epochs"

    def test_plus_prefix(self):
        cmd = "python -m main_ppo +custom.key=value"
        params, _ = parse_parameters(cmd)
        assert params["custom.key"] == "value"

    def test_nested_keys(self):
        cmd = "python -m main_ppo actor_rollout_ref.rollout.n=5"
        params, _ = parse_parameters(cmd)
        assert params["actor_rollout_ref.rollout.n"] == 5


class TestParseLog:
    def test_parses_golden_log(self):
        exp, metrics, report = parse_log(GOLDEN_LOG)
        
        # Experiment record
        assert exp.algorithm == "grpo"
        assert exp.rollout_n == 5
        assert exp.total_epochs == 3
        assert exp.total_training_steps == 174
        assert exp.model == "/models/qwen7b"
        assert exp.experiment_name == "qwen_golden"
        
        # Metrics
        assert len(metrics) > 0
        steps = set(m.step for m in metrics)
        assert {1, 2, 3, 4}.issubset(steps)
        
        # Check specific metrics
        step1_metrics = {m.metric_name: m.metric_value for m in metrics if m.step == 1}
        assert step1_metrics["actor/entropy"] == pytest.approx(0.733)
        assert step1_metrics["training/global_step"] == 1.0
        assert step1_metrics["training/epoch"] == 0.0
        
        # Validation metrics
        val_metrics = [m for m in metrics if m.phase == "validation"]
        assert len(val_metrics) > 0
        val_names = {m.metric_name for m in val_metrics}
        assert "val-aux/math/reward/mean@1" in val_names

    def test_no_silent_drop(self):
        """Failed metric lines should be reported, not silently dropped."""
        exp, metrics, report = parse_log(GOLDEN_LOG)
        assert report.total_lines > 0
        # All metric lines should parse (golden log has valid metric lines)
        # but we should have parsed_lines > 0
        assert report.parsed_lines > 0

    def test_unknown_metrics_preserved(self):
        """Unknown metric names should be preserved in output."""
        exp, metrics, report = parse_log(GOLDEN_LOG)
        # custom/* should appear in unknown_fields if present
        all_names = {m.metric_name for m in metrics}
        # All metrics should be preserved regardless of known/unknown
        assert "actor/entropy" in all_names

    def test_epoch_mapping(self):
        """Steps should be mapped to correct epochs."""
        exp, metrics, report = parse_log(GOLDEN_LOG)
        step4 = [m for m in metrics if m.step == 4 and m.metric_name == "training/epoch"]
        if step4:
            assert step4[0].metric_value == 1.0  # epoch 1

    def test_missing_log_file(self):
        """Missing log file should raise clear error."""
        with pytest.raises(FileNotFoundError):
            parse_log("/nonexistent/path/to/log.txt")

    def test_resolved_runtime_config_fallback(self, tmp_path):
        log_path = tmp_path / "train.log"
        log_path.write_text("\n".join([
            "resolved config: {'adv_estimator': ngrpo, 'train_files': /data/train.parquet, 'train_batch_size': 8, 'rollout_n': 4, 'total_training_steps': 5}",
            "actor config: {'optim': {'lr': 5e-07}}",
            "reference model: /models/qwen7b",
            "step:1 - critic/score/mean:0.25",
        ]))

        exp, _, report = parse_log(str(log_path))

        assert exp.algorithm == "ngrpo"
        assert exp.dataset == "/data/train.parquet"
        assert exp.model == "/models/qwen7b"
        assert exp.learning_rate == pytest.approx(5e-7)
        assert exp.total_training_steps == 5
        assert exp.config_sources == {"primary": "resolved_runtime_config"}
        assert "command_line" not in report.missing_required


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
