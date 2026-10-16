import json

from rl_diagnostics.experiment_store import ExperimentStore
from rl_diagnostics.notebook_view import render_experiment_console


def _make_run(root, name, algorithm, reward):
    run_dir = root / name
    rollout_dir = run_dir / "rollouts"
    rollout_dir.mkdir(parents=True)
    (run_dir / "train.log").write_text("\n".join([
        f"python -m verl.trainer.main_ppo trainer.experiment_name={name} algorithm.adv_estimator={algorithm}",
        f"step:1 - critic/score/mean:{reward} - actor/kl_loss:0.01 - actor/grad_norm:1.0",
    ]))
    records = [
        {"input": "q", "output": "a", "score": 1.0, "acc": True, "uid": "g1"},
        {"input": "q", "output": "b", "score": 0.0, "acc": False, "uid": "g1"},
    ]
    (rollout_dir / "1.jsonl").write_text("\n".join(json.dumps(item) for item in records))


def test_notebook_console_renders_diagnostics_and_comparison(tmp_path):
    runs_root = tmp_path / "runs"
    _make_run(runs_root, "baseline", "grpo", 0.2)
    _make_run(runs_root, "candidate", "ngrpo", 0.4)
    store = ExperimentStore(runs_root, tmp_path / "state")
    experiments = store.refresh_all()

    result = render_experiment_console(
        store,
        selected_id=experiments[0]["experiment_id"],
        compare_id=experiments[1]["experiment_id"],
        refresh=False,
    )

    assert "Rollout 质量" in result
    assert "稳定性规则" in result
    assert "实验参数与指标对比" in result
    assert "algorithm.adv_estimator" in result
    assert '<details class="rule' in result
    assert "最新 Rollout 证据" in result


def test_notebook_console_handles_empty_root(tmp_path):
    store = ExperimentStore(tmp_path / "runs", tmp_path / "state")

    result = render_experiment_console(store)

    assert "没有发现实验" in result
