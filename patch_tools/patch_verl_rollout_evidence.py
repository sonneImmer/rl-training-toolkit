#!/usr/bin/env python3
"""Patch a VERL checkout to persist and print rollout learning-signal evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


MARKER = "zkx-rollout-evidence-patch-v5"
COMPATIBLE_BACKUP_MARKERS = {MARKER, "zkx-rollout-evidence-patch-v4"}

SUMMARY_METHOD = '''    def _log_rl_diagnostic_summary(self, batch: DataProto) -> None:
        """Print one machine-readable learning-signal summary for the current training step."""
        try:
            score_tensor = batch.batch.get("token_level_scores")
            if score_tensor is None:
                return
            scores = score_tensor.sum(-1).detach().float().cpu().tolist()
            response_mask = batch.batch.get("response_mask")
            advantage_tensor = batch.batch.get("advantages")
            sequence_advantages = []
            response_mask_tokens = 0
            policy_signal_tokens = 0
            if advantage_tensor is not None:
                if response_mask is not None:
                    mask = response_mask.bool()
                    sequence_advantages = (
                        (advantage_tensor * response_mask).sum(-1)
                        / response_mask.sum(-1).clamp_min(1)
                    ).detach().float().cpu().tolist()
                    response_mask_tokens = int(response_mask.sum().detach().cpu().item())
                    policy_signal_tokens = int(((advantage_tensor.abs() > 1e-8) & mask).sum().detach().cpu().item())
                else:
                    sequence_advantages = advantage_tensor.mean(-1).detach().float().cpu().tolist()

            rollout_n = max(int(self.config.actor_rollout_ref.rollout.n), 1)
            raw_uids = batch.non_tensor_batch.get("uid")
            uid_values = raw_uids.tolist() if raw_uids is not None else [""] * len(scores)
            correctness_values = None
            for field in ("acc", "correctness", "correct"):
                raw_values = batch.non_tensor_batch.get(field)
                if raw_values is not None:
                    correctness_values = raw_values.tolist()
                    break

            groups = {}
            for index, score in enumerate(scores):
                raw_uid = uid_values[index] if index < len(uid_values) else ""
                uid = str(raw_uid).strip()
                group_key = uid if uid else f"fallback-{index // rollout_n}"
                group = groups.setdefault(group_key, {"scores": [], "advantages": [], "correctness": []})
                group["scores"].append(float(score))
                if index < len(sequence_advantages):
                    group["advantages"].append(float(sequence_advantages[index]))
                if correctness_values is not None and index < len(correctness_values):
                    value = correctness_values[index]
                    if isinstance(value, (bool, np.bool_)):
                        group["correctness"].append(bool(value))
                    elif str(value).strip().lower() in {"true", "1", "yes"}:
                        group["correctness"].append(True)
                    elif str(value).strip().lower() in {"false", "0", "no"}:
                        group["correctness"].append(False)

            grouped = list(groups.values())
            constant_groups = sum(max(group["scores"]) - min(group["scores"]) <= 1e-8 for group in grouped)
            effective_groups = len(grouped) - constant_groups
            all_positive_groups = sum(all(value > 0 for value in group["scores"]) for group in grouped)
            all_non_positive_groups = sum(all(value <= 0 for value in group["scores"]) for group in grouped)
            mixed_sign_groups = len(grouped) - all_positive_groups - all_non_positive_groups
            correctness_groups = [
                group for group in grouped if len(group["correctness"]) == len(group["scores"])
            ]
            all_correct_groups = sum(all(group["correctness"]) for group in correctness_groups)
            all_wrong_groups = sum(not any(group["correctness"]) for group in correctness_groups)
            mixed_correctness_groups = len(correctness_groups) - all_correct_groups - all_wrong_groups
            positive_advantages = sum(value > 1e-8 for value in sequence_advantages)
            negative_advantages = sum(value < -1e-8 for value in sequence_advantages)
            zero_advantages = len(sequence_advantages) - positive_advantages - negative_advantages
            group_count = len(grouped)

            summary = {
                "step": int(self.global_steps),
                "algorithm": str(self.config.algorithm.adv_estimator),
                "actor_update_executed": bool(self.config.trainer.critic_warmup <= self.global_steps),
                "responses": len(scores),
                "groups": group_count,
                "reward_mean": float(np.mean(scores)) if scores else None,
                "reward_std": float(np.std(scores)) if scores else None,
                "reward_min": float(min(scores)) if scores else None,
                "reward_max": float(max(scores)) if scores else None,
                "effective_groups": effective_groups,
                "constant_reward_groups": constant_groups,
                "effective_group_ratio": effective_groups / group_count if group_count else None,
                "all_positive_groups": all_positive_groups,
                "all_non_positive_groups": all_non_positive_groups,
                "mixed_sign_groups": mixed_sign_groups,
                "correctness_groups": len(correctness_groups),
                "all_correct_groups": all_correct_groups,
                "all_wrong_groups": all_wrong_groups,
                "mixed_correctness_groups": mixed_correctness_groups,
                "advantage_recorded": len(sequence_advantages),
                "advantage_positive": positive_advantages,
                "advantage_negative": negative_advantages,
                "advantage_zero": zero_advantages,
                "advantage_mean": float(np.mean(sequence_advantages)) if sequence_advantages else None,
                "response_mask_tokens": response_mask_tokens,
                "policy_signal_tokens": policy_signal_tokens,
            }
            print(f"[RL_DIAG_SUMMARY] {json.dumps(summary, ensure_ascii=False, sort_keys=True)}", flush=True)
        except Exception as error:
            print(
                f"[RL_DIAG_SUMMARY_ERROR] step={self.global_steps} "
                f"error={type(error).__name__}: {error}",
                flush=True,
            )
'''


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
        if manifest.get("marker") not in COMPATIBLE_BACKUP_MARKERS:
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
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def patch_checks(files: dict[str, Path]) -> dict[str, bool]:
    text = files["ray_trainer"].read_text(encoding="utf-8")
    return {
        "rollout_id": '"rollout_id": list(range(n))' in text,
        "group_metadata": 'reward_extra_infos_to_dump["uid"]' in text,
        "rollout_advantage": 'reward_extra_infos_to_dump["advantage"]' in text,
        "training_selection": 'reward_extra_infos_to_dump["update_candidate"]' in text,
        "actor_update_observed": 'reward_extra_infos_to_dump["actor_update_executed"]' in text,
        "console_summary": (
            "def _log_rl_diagnostic_summary(" in text
            and "self._log_rl_diagnostic_summary(batch)" in text
        ),
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
            '            "score": scores,\n'
            '            "step": [self.global_steps] * n,\n'
        )
        replace_once(path, anchor, anchor + '            "rollout_id": list(range(n)),\n')

    if not checks["rollout_advantage"]:
        score_anchor = '            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()\n'
        advantage_code = (
            score_anchor
            + '            advantages = (\n'
            + '                (batch.batch["advantages"] * batch.batch["response_mask"]).sum(-1)\n'
            + '                / batch.batch["response_mask"].sum(-1).clamp_min(1)\n'
            + '            ).detach().cpu().tolist()\n'
        )
        replace_once(path, score_anchor, advantage_code)

    if not checks["training_selection"] or not checks["actor_update_observed"]:
        selection_anchor = (
            '            advantages = (\n'
            '                (batch.batch["advantages"] * batch.batch["response_mask"]).sum(-1)\n'
            '                / batch.batch["response_mask"].sum(-1).clamp_min(1)\n'
            '            ).detach().cpu().tolist()\n'
        )
        selection_code = (
            selection_anchor
            + '            response_mask_tokens = batch.batch["response_mask"].sum(-1).detach().cpu().tolist()\n'
            + '            policy_signal_tokens = (\n'
            + '                (batch.batch["advantages"].abs() > 1e-8) & batch.batch["response_mask"].bool()\n'
            + '            ).sum(-1).detach().cpu().tolist()\n'
            + '            actor_update_enabled = self.config.trainer.critic_warmup <= self.global_steps\n'
            + '            update_candidate = [bool(actor_update_enabled and tokens > 0) for tokens in response_mask_tokens]\n'
            + '            actor_update_executed = [bool(actor_update_enabled)] * len(response_mask_tokens)\n'
        )
        replace_once(path, selection_anchor, selection_code)

        text = path.read_text(encoding="utf-8")
        metadata_anchor = '            reward_extra_infos_to_dump["data_source"] = data_source_list\n'
        evidence_fields = (
            '            reward_extra_infos_to_dump["advantage"] = advantages\n'
            '            reward_extra_infos_to_dump["selected_for_update"] = update_candidate\n'
            '            reward_extra_infos_to_dump["update_candidate"] = update_candidate\n'
            '            reward_extra_infos_to_dump["actor_update_executed"] = actor_update_executed\n'
            '            reward_extra_infos_to_dump["response_mask_tokens"] = response_mask_tokens\n'
            '            reward_extra_infos_to_dump["policy_signal_tokens"] = policy_signal_tokens\n'
            '            reward_extra_infos_to_dump["adv_estimator"] = '
            '[str(self.config.algorithm.adv_estimator)] * len(inputs)\n'
        )
        if metadata_anchor in text:
            replace_once(path, metadata_anchor, metadata_anchor + evidence_fields)
        else:
            reward_anchor = '            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()\n'
            metadata_code = (
                '            uid_list = batch.non_tensor_batch.get("uid", np.array([""] * len(inputs))).tolist() '
                'if "uid" in batch.non_tensor_batch else [""] * len(inputs)\n'
                '            data_source_list = batch.non_tensor_batch.get("data_source", np.array([""] * len(inputs))).tolist() '
                'if "data_source" in batch.non_tensor_batch else [""] * len(inputs)\n\n'
                + reward_anchor
                + '            reward_extra_infos_to_dump["uid"] = uid_list\n'
                + '            reward_extra_infos_to_dump["data_source"] = data_source_list\n'
                + evidence_fields
            )
            replace_once(path, reward_anchor, metadata_code)

    if not checks["console_summary"]:
        method_anchor = "    def _log_rollout_data(\n"
        replace_once(path, method_anchor, SUMMARY_METHOD + "\n" + method_anchor)
        call_anchor = "                    # Log rollout generations if enabled\n"
        replace_once(
            path,
            call_anchor,
            "                    self._log_rl_diagnostic_summary(batch)\n\n" + call_anchor,
        )

    manifest_path = backup_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["marker"] = MARKER
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
