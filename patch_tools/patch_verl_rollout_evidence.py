#!/usr/bin/env python3
"""Patch a VERL checkout to persist rollout group metadata and sequence advantages."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


MARKER = "zkx-rollout-evidence-patch-v3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one patch anchor, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def target_files(verl_root: Path) -> dict[str, Path]:
    return {"ray_trainer": verl_root / "verl/trainer/ppo/ray_trainer.py"}


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
    text = files["ray_trainer"].read_text(encoding="utf-8")
    return {
        "rollout_id": "\"rollout_id\": list(range(n))" in text,
        "group_metadata": "reward_extra_infos_to_dump[\"uid\"]" in text,
        "rollout_advantage": "reward_extra_infos_to_dump[\"advantage\"]" in text,
        "training_selection": "reward_extra_infos_to_dump[\"selected_for_update\"]" in text,
    }


def apply_patch(verl_root: Path, backup_dir: Path) -> None:
    files = target_files(verl_root)
    checks = patch_checks(files)
    if all(checks.values()):
        print("Rollout evidence patch already applied")
        return
    backup(files, backup_dir)
    path = files["ray_trainer"]

    if not checks["rollout_id"]:
        anchor = (
            "            \"score\": scores,\n"
            "            \"step\": [self.global_steps] * n,\n"
        )
        replace_once(path, anchor, anchor + "            \"rollout_id\": list(range(n)),\n")

    if not checks["rollout_advantage"]:
        score_anchor = "            scores = batch.batch[\"token_level_scores\"].sum(-1).cpu().tolist()\n"
        advantage_code = (
            score_anchor
            + "            advantages = (\n"
            + "                (batch.batch[\"advantages\"] * batch.batch[\"response_mask\"]).sum(-1)\n"
            + "                / batch.batch[\"response_mask\"].sum(-1).clamp_min(1)\n"
            + "            ).detach().cpu().tolist()\n"
        )
        replace_once(path, score_anchor, advantage_code)

    if not checks["training_selection"]:
        selection_anchor = (
            "            advantages = (\n"
            "                (batch.batch[\"advantages\"] * batch.batch[\"response_mask\"]).sum(-1)\n"
            "                / batch.batch[\"response_mask\"].sum(-1).clamp_min(1)\n"
            "            ).detach().cpu().tolist()\n"
        )
        selection_code = (
            selection_anchor
            + "            response_mask_tokens = batch.batch[\"response_mask\"].sum(-1).detach().cpu().tolist()\n"
            + "            policy_signal_tokens = (\n"
            + "                (batch.batch[\"advantages\"].abs() > 1e-8) & batch.batch[\"response_mask\"].bool()\n"
            + "            ).sum(-1).detach().cpu().tolist()\n"
            + "            actor_update_enabled = self.config.trainer.critic_warmup <= self.global_steps\n"
            + "            selected_for_update = [bool(actor_update_enabled and tokens > 0) for tokens in response_mask_tokens]\n"
        )
        replace_once(path, selection_anchor, selection_code)

        text = path.read_text(encoding="utf-8")
        metadata_anchor = "            reward_extra_infos_to_dump[\"data_source\"] = data_source_list\n"
        evidence_fields = (
            "            reward_extra_infos_to_dump[\"advantage\"] = advantages\n"
            "            reward_extra_infos_to_dump[\"selected_for_update\"] = selected_for_update\n"
            "            reward_extra_infos_to_dump[\"response_mask_tokens\"] = response_mask_tokens\n"
            "            reward_extra_infos_to_dump[\"policy_signal_tokens\"] = policy_signal_tokens\n"
            "            reward_extra_infos_to_dump[\"adv_estimator\"] = "
            "[str(self.config.algorithm.adv_estimator)] * len(inputs)\n"
        )
        if metadata_anchor in text:
            replace_once(path, metadata_anchor, metadata_anchor + evidence_fields)
        else:
            reward_anchor = "            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()\n"
            metadata_code = (
                "            uid_list = batch.non_tensor_batch.get(\"uid\", np.array([\"\"] * len(inputs))).tolist() "
                "if \"uid\" in batch.non_tensor_batch else [\"\"] * len(inputs)\n"
                "            data_source_list = batch.non_tensor_batch.get(\"data_source\", np.array([\"\"] * len(inputs))).tolist() "
                "if \"data_source\" in batch.non_tensor_batch else [\"\"] * len(inputs)\n\n"
                + reward_anchor
                + "            reward_extra_infos_to_dump[\"uid\"] = uid_list\n"
                + "            reward_extra_infos_to_dump[\"data_source\"] = data_source_list\n"
                + evidence_fields
            )
            replace_once(path, reward_anchor, metadata_code)

    manifest_path = backup_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["ray_trainer"]["after_sha256"] = sha256(path)
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
    parser.add_argument("--backup-dir", type=Path, default=Path("/workspace/verl_rollout_evidence_backup"))
    args = parser.parse_args()
    if args.command == "apply":
        apply_patch(args.verl_root.resolve(), args.backup_dir.resolve())
    elif args.command == "check":
        check(args.verl_root.resolve())
    else:
        restore(args.backup_dir.resolve())


if __name__ == "__main__":
    main()
