#!/usr/bin/env python3
"""Patch a verl 0.7 checkout with the predecessor NGRPO advantage formula."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


MARKER = "zkx-ngrpo-runtime-patch-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one patch anchor, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def target_files(verl_root: Path) -> dict[str, Path]:
    return {
        "core_algos": verl_root / "verl/trainer/ppo/core_algos.py",
        "algorithm_config": verl_root / "verl/trainer/config/algorithm.py",
        "ppo_yaml": verl_root / "verl/trainer/config/ppo_trainer.yaml",
    }


def backup(files: dict[str, Path], backup_dir: Path) -> None:
    manifest_path = backup_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("marker") != MARKER:
            raise RuntimeError(f"{backup_dir}: incompatible backup manifest")
        return
    backup_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"marker": MARKER, "files": {}}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        target = backup_dir / f"{name}.bak"
        shutil.copy2(path, target)
        manifest["files"][name] = {
            "path": str(path),
            "before_sha256": sha256(path),
            "backup": str(target),
        }
    (backup_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def patch_checks(files: dict[str, Path]) -> dict[str, bool]:
    return {
        "registered_ngrpo": '@register_adv_est("ngrpo")' in files["core_algos"].read_text(),
        "config_max_reward": "max_reward: float = 1.0" in files["algorithm_config"].read_text(),
        "yaml_max_reward": "max_reward: 1.0" in files["ppo_yaml"].read_text(),
    }


def apply_patch(verl_root: Path, backup_dir: Path) -> None:
    files = target_files(verl_root)
    checks = patch_checks(files)
    if all(checks.values()):
        print("NGRPO patch already applied")
        return
    backup(files, backup_dir)

    grpo_anchor = "@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est(\"grpo\")\n"
    ngrpo_code = f'''# {MARKER}\n@register_adv_est("ngrpo")\ndef compute_ngrpo_outcome_advantage(\n    token_level_rewards: torch.Tensor,\n    response_mask: torch.Tensor,\n    index: np.ndarray,\n    epsilon: float = 1e-6,\n    config: Optional[AlgoConfig] = None,\n    **kwargs,\n) -> tuple[torch.Tensor, torch.Tensor]:\n    """NGRPO outcome advantage using predecessor virtual max-reward samples."""\n    if config is None:\n        raise ValueError("NGRPO requires algorithm config")\n    max_reward = float(config.get("max_reward", 1.0))\n    add_max_reward_num = int(config.get("add_max_reward_num", 1))\n    if add_max_reward_num < 1:\n        raise ValueError("algorithm.add_max_reward_num must be >= 1")\n\n    scores = token_level_rewards.sum(dim=-1)\n    group_scores = defaultdict(list)\n    group_mean = {{}}\n    group_std = {{}}\n    with torch.no_grad():\n        for sample_index in range(scores.shape[0]):\n            group_scores[index[sample_index]].append(scores[sample_index])\n        for group_id, values in group_scores.items():\n            values_tensor = torch.stack(values)\n            virtual_scores = torch.full(\n                (add_max_reward_num,),\n                max_reward,\n                dtype=values_tensor.dtype,\n                device=values_tensor.device,\n            )\n            augmented = torch.cat((values_tensor, virtual_scores))\n            group_mean[group_id] = augmented.mean()\n            group_std[group_id] = augmented.std()\n        normalized = torch.stack([\n            (scores[sample_index] - group_mean[index[sample_index]])\n            / (group_std[index[sample_index]] + epsilon)\n            for sample_index in range(scores.shape[0])\n        ])\n        advantages = normalized.unsqueeze(-1) * response_mask\n    return advantages, advantages\n\n\n'''
    if not checks["registered_ngrpo"]:
        replace_once(files["core_algos"], grpo_anchor, ngrpo_code + grpo_anchor)

    config_anchor = '    norm_adv_by_std_in_grpo: bool = True\n'
    config_fields = (
        '    norm_adv_by_std_in_grpo: bool = True\n'
        '    max_reward: float = 1.0\n'
        '    add_max_reward_num: int = 1\n'
    )
    if not checks["config_max_reward"]:
        replace_once(files["algorithm_config"], config_anchor, config_fields)

    yaml_anchor = "  norm_adv_by_std_in_grpo: True\n"
    yaml_fields = (
        "  norm_adv_by_std_in_grpo: True\n\n"
        "  # NGRPO virtual max-reward settings\n"
        "  max_reward: 1.0\n"
        "  add_max_reward_num: 1\n"
    )
    if not checks["yaml_max_reward"]:
        replace_once(files["ppo_yaml"], yaml_anchor, yaml_fields)

    manifest_path = backup_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, path in files.items():
        if name in manifest["files"]:
            manifest["files"][name]["after_sha256"] = sha256(path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Applied {MARKER}; backup: {backup_dir}")


def restore(backup_dir: Path) -> None:
    manifest = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        shutil.copy2(info["backup"], info["path"])
        print(f"restored {name}: {info['path']}")


def check(verl_root: Path) -> None:
    files = target_files(verl_root)
    checks = patch_checks(files)
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("apply", "check", "restore"))
    parser.add_argument("--verl-root", type=Path, default=Path("/workspace/verl"))
    parser.add_argument("--backup-dir", type=Path, default=Path("/workspace/verl_ngrpo_backup"))
    args = parser.parse_args()
    if args.command == "apply":
        apply_patch(args.verl_root.resolve(), args.backup_dir.resolve())
    elif args.command == "check":
        check(args.verl_root.resolve())
    else:
        restore(args.backup_dir.resolve())


if __name__ == "__main__":
    main()
