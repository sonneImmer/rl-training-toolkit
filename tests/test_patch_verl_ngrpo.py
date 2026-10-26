import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "patch_tools" / "patch_verl_ngrpo.py"
SPEC = importlib.util.spec_from_file_location("patch_verl_ngrpo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _make_verl_root(tmp_path):
    root = tmp_path / "verl-root"
    files = MODULE.target_files(root)
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    files["core_algos"].write_text(
        '@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")\n'
    )
    files["algorithm_config"].write_text("    norm_adv_by_std_in_grpo: bool = True\n")
    files["ppo_yaml"].write_text("  norm_adv_by_std_in_grpo: True\n")
    return root


def test_minimal_patch_does_not_require_ray_trainer(tmp_path):
    root = _make_verl_root(tmp_path)
    backup = tmp_path / "backup"

    MODULE.apply_patch(root, backup)
    MODULE.check(root)
    MODULE.apply_patch(root, backup)

    assert all(MODULE.patch_checks(MODULE.target_files(root)).values())
    manifest = json.loads((backup / "manifest.json").read_text())
    assert "ray_trainer" not in manifest["files"]


def test_partial_old_install_is_accepted(tmp_path):
    root = _make_verl_root(tmp_path)
    backup = tmp_path / "backup"
    MODULE.apply_patch(root, backup)

    old_manifest = json.loads((backup / "manifest.json").read_text())
    old_manifest["files"]["ray_trainer"] = {
        "path": "/workspace/verl/verl/trainer/ppo/ray_trainer.py",
        "backup": "/workspace/verl_ngrpo_backup/ray_trainer.bak",
    }
    (backup / "manifest.json").write_text(json.dumps(old_manifest))

    MODULE.apply_patch(root, backup)
    MODULE.check(root)


def test_rollout_evidence_patcher_records_advantage(tmp_path):
    script = Path(__file__).parents[1] / "patch_tools" / "patch_verl_rollout_evidence.py"
    spec = importlib.util.spec_from_file_location("patch_verl_rollout_evidence", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    root = tmp_path / "verl-root"
    path = root / "verl/trainer/ppo/ray_trainer.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        '            "score": scores,\n'
        '            "step": [self.global_steps] * n,\n'
        '            "rollout_id": list(range(n)),\n'
        '            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()\n'
        '            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()\n'
        '            reward_extra_infos_to_dump["uid"] = uid_list\n'
        '            reward_extra_infos_to_dump["data_source"] = data_source_list\n'
    )
    backup = tmp_path / "backup"

    module.apply_patch(root, backup)
    module.check(root)
    patched = path.read_text()
    assert 'reward_extra_infos_to_dump["advantage"] = advantages' in patched
    assert 'batch.batch["advantages"]' in patched
