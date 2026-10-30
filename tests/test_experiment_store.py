import json

from rl_diagnostics.experiment_store import ExperimentStore


def test_experiment_object_is_persisted(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "grpo-test" / "run-001"
    rollout_dir = run_dir / "rollouts"
    rollout_dir.mkdir(parents=True)
    log = "\n".join([
        "python -m verl.trainer.main_ppo trainer.experiment_name=grpo-test algorithm.adv_estimator=grpo trainer.total_training_steps=2",
        "step:1 - actor/grad_norm:1.0 - actor/kl_loss:0.01 - critic/score/mean:0.0 - response_length/clip_ratio:0.0",
        "step:2 - actor/grad_norm:1.1 - actor/kl_loss:0.02 - critic/score/mean:0.5 - response_length/clip_ratio:0.0",
        "Training Progress: 100%|##########| 2/2",
        "Traceback (most recent call last): cleanup failure",
    ])
    (run_dir / "train.log").write_text(log)
    records = [
        {"input": "q", "output": "a", "score": 1.0, "acc": True, "uid": "g1"},
        {"input": "q", "output": "b", "score": -1.0, "acc": False, "uid": "g1"},
    ]
    (rollout_dir / "2.jsonl").write_text("\n".join(json.dumps(item) for item in records))

    state_dir = tmp_path / "state"
    store = ExperimentStore(runs_root, state_dir)
    snapshots = store.refresh_all()

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["status"] == "completed_with_errors"
    assert snapshot["progress"] == {"current_step": 2, "total_steps": 2, "ratio": 1.0}
    assert snapshot["latest_metrics"]["has_correctness"] is True
    object_dir = state_dir / "experiments" / snapshot["experiment_id"]
    assert (object_dir / "manifest.json").is_file()
    assert (object_dir / "snapshot.json").is_file()
    assert (object_dir / "final_snapshot.json").is_file()
    assert (object_dir / "events.jsonl").is_file()
    error_label = next(item for item in snapshot["labels"] if item["name"] == "post_training_error")
    assert "cleanup failure" in error_label["observation"]


def test_rollout_evidence_can_be_loaded(tmp_path):
    run_dir = tmp_path / "runs" / "run-001"
    rollout_dir = run_dir / "rollouts"
    rollout_dir.mkdir(parents=True)
    (run_dir / "train.log").write_text("step:1 - critic/score/mean:1.0\n")
    (rollout_dir / "1.jsonl").write_text(json.dumps({"output": "answer", "score": 1.0}) + "\n")
    store = ExperimentStore(tmp_path / "runs", tmp_path / "state", stale_seconds=999999)
    snapshot = store.refresh_all()[0]

    records = store.get_rollouts(snapshot["experiment_id"], step=1)

    assert records[0]["output"] == "answer"


def test_standard_diagnostics_are_persisted(tmp_path):
    run_dir = tmp_path / "runs" / "run-001"
    rollout_dir = run_dir / "rollouts"
    rollout_dir.mkdir(parents=True)
    (run_dir / "train.log").write_text(
        "step:1 - actor/entropy:1.0 - actor/kl_loss:0.01 - actor/grad_norm:1.0\n"
    )
    records = [
        {"input": "q", "output": "a", "score": 1.0, "acc": True, "uid": "g1"},
        {"input": "q", "output": "b", "score": 0.0, "acc": False, "uid": "g1"},
    ]
    (rollout_dir / "1.jsonl").write_text("\n".join(json.dumps(item) for item in records))

    snapshot = ExperimentStore(tmp_path / "runs", tmp_path / "state").refresh_all()[0]

    assert snapshot["latest_metrics"]["effective_group_ratio"] == 1.0
    assert snapshot["latest_metrics"]["rollout_usable_ratio"] == 1.0
    assert snapshot["diagnostics"]["quality_by_step"][0]["counts"] == {"usable": 2}
    evaluations = snapshot["diagnostics"]["stability"]["evaluations"]
    assert any(item["status"] == "not_evaluated" for item in evaluations)



def test_short_experiment_gets_low_confidence_early_observations(tmp_path):
    run_dir = tmp_path / "runs" / "run-early"
    run_dir.mkdir(parents=True)
    (run_dir / "train.log").write_text("\n".join([
        "step:1 - actor/entropy:1.0 - actor/kl_loss:0.01 - actor/grad_norm:1.0",
        "step:2 - actor/entropy:0.9 - actor/kl_loss:0.02 - actor/grad_norm:1.1",
        "step:3 - actor/entropy:0.8 - actor/kl_loss:0.03 - actor/grad_norm:1.2",
    ]))

    snapshot = ExperimentStore(tmp_path / "runs", tmp_path / "state").refresh_all()[0]

    evaluations = snapshot["diagnostics"]["stability"]["evaluations"]
    early = [item for item in evaluations if item["status"].startswith("early_")]
    assert early
    assert all(item["confidence"] == "low" for item in early)
    assert all(item["window_steps"] for item in early)
    assert all(item["sample_count"] == 3 for item in early)
    assert all(item["required_samples"] > item["sample_count"] for item in early)
    assert all("正式判断" in item["observation"] for item in early)
    unavailable = [item for item in evaluations if item["status"] == "not_evaluated"]
    assert all(item["observation"] and item["decision_reason"] for item in unavailable)
def test_experiment_comparison_reports_parameter_changes(tmp_path):
    runs_root = tmp_path / "runs"
    for name, algorithm, learning_rate, reward in (
        ("baseline", "grpo", "1e-6", 0.2),
        ("candidate", "ngrpo", "2e-6", 0.4),
    ):
        run_dir = runs_root / name
        run_dir.mkdir(parents=True)
        (run_dir / "train.log").write_text("\n".join([
            f"python -m verl.trainer.main_ppo algorithm.adv_estimator={algorithm} actor_rollout_ref.actor.optim.lr={learning_rate}",
            f"step:1 - critic/score/mean:{reward}",
        ]))

    store = ExperimentStore(runs_root, tmp_path / "state")
    snapshots = store.refresh_all()
    result = store.compare_experiments([item["experiment_id"] for item in snapshots])

    assert result is not None
    changed = {row["key"] for row in result["config_rows"] if row["changed"]}
    assert "algorithm.adv_estimator" in changed
    assert "actor_rollout_ref.actor.optim.lr" in changed
    reward_row = next(row for row in result["latest_metric_rows"] if row["key"] == "reward_mean")
    assert abs(reward_row["delta"]) == 0.2


def test_run_manifest_controls_comparison_readiness(tmp_path):
    run_dir = tmp_path / "runs" / "run-001"
    run_dir.mkdir(parents=True)
    (run_dir / "train.log").write_text("\n".join([
        "python -m verl.trainer.main_ppo algorithm.adv_estimator=grpo trainer.total_training_steps=2",
        "step:1 - critic/score/mean:0.5",
    ]))
    manifest = {
        "model": {"id": "/models/qwen7b"},
        "dataset": {"id": "/data/finance/train.parquet"},
        "reward": {"definition": "risk classification reward", "min": -4, "max": 4},
        "training": {"seed": 42, "total_steps": 2},
        "environment": {"image": "openpangu:v1", "code_revision": "abc123"},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))

    snapshot = ExperimentStore(tmp_path / "runs", tmp_path / "state").refresh_all()[0]

    assert snapshot["run_manifest"] == manifest
    assert snapshot["comparison_readiness"]["eligible"] is True
    indexed = ExperimentStore(tmp_path / "runs", tmp_path / "state").list_experiments()[0]
    assert indexed["comparison_readiness"]["eligible"] is True
    persisted = json.loads((tmp_path / "state" / "experiments" / snapshot["experiment_id"] / "manifest.json").read_text())
    assert persisted["comparison_readiness"]["status"] == "ready"


def test_independent_evaluations_are_attached_to_experiment(tmp_path):
    run_dir = tmp_path / "runs" / "run-001"
    evaluation_dir = run_dir / "evaluations" / "checkpoint-step-5"
    evaluation_dir.mkdir(parents=True)
    (run_dir / "train.log").write_text("step:5 - critic/score/mean:1.0\n")
    (evaluation_dir / "metrics.json").write_text(json.dumps({
        "samples": 20,
        "risk": {"accuracy": 0.9, "recall": 0.8, "false_negative": 2},
        "per_typology": {"FAN_OUT": {"samples": 5, "correct": 4, "accuracy": 0.8}},
    }))
    (evaluation_dir / "evaluation_manifest.json").write_text(json.dumps({
        "evaluation_id": "checkpoint-step-5", "checkpoint_step": 5, "split": "test",
    }))

    snapshot = ExperimentStore(tmp_path / "runs", tmp_path / "state").refresh_all()[0]

    assert snapshot["evaluations"][0]["checkpoint_step"] == 5
    assert snapshot["evaluations"][0]["summary"]["risk_accuracy"] == 0.9
    assert snapshot["evaluations"][0]["per_class"]["FAN_OUT"]["accuracy"] == 0.8
    indexed = ExperimentStore(tmp_path / "runs", tmp_path / "state").list_experiments()[0]
    assert indexed["evaluation_count"] == 1
    assert snapshot["evaluation_alerts"] == []


def test_independent_evaluation_detects_overall_up_class_down(tmp_path):
    run_dir = tmp_path / "runs" / "run-001"
    run_dir.mkdir(parents=True)
    (run_dir / "train.log").write_text("step:2 - critic/score/mean:1.0\n")
    for step, overall, fan_out in ((1, 0.90, 0.95), (2, 0.96, 0.80)):
        evaluation_dir = run_dir / "evaluations" / f"step-{step}"
        evaluation_dir.mkdir(parents=True)
        (evaluation_dir / "metrics.json").write_text(json.dumps({
            "risk": {"accuracy": overall},
            "per_typology": {"FAN_OUT": {"samples": 20, "correct": int(20 * fan_out), "accuracy": fan_out}},
        }))
        (evaluation_dir / "evaluation_manifest.json").write_text(json.dumps({
            "evaluation_id": f"step-{step}", "checkpoint_step": step,
        }))

    snapshot = ExperimentStore(tmp_path / "runs", tmp_path / "state").refresh_all()[0]

    alert = snapshot["evaluation_alerts"][0]
    assert alert["type"] == "overall_up_class_down"
    assert alert["class_name"] == "FAN_OUT"
    assert alert["overall_after"] == 0.96
    assert alert["class_after"] == 0.80


def test_dashboard_summary_includes_evaluation_count(tmp_path):
    from rl_diagnostics.dashboard import _summary

    item = {"evaluations": [{"summary": {"accuracy": 0.75}}], "labels": []}

    summary = _summary(item)
    assert summary["evaluation_count"] == 1
    assert summary["latest_evaluation"]["accuracy"] == 0.75

    cached = _summary({"evaluation_count": 3, "latest_evaluation": {"accuracy": 0.8}, "labels": []})
    assert cached["evaluation_count"] == 3
    assert cached["latest_evaluation"]["accuracy"] == 0.8
