#!/usr/bin/env python3
"""Add a reversible gradient-sync mode switch to the legacy VERL DP actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


MARKER = "zkx-gradient-sync-memory-case-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one patch anchor, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def target_file(verl_root: Path) -> Path:
    return verl_root / "verl/workers/actor/dp_actor.py"


def backup(path: Path, backup_dir: Path) -> None:
    manifest_path = backup_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("marker") != MARKER:
            raise RuntimeError(f"{backup_dir}: incompatible backup manifest")
        return
    backup_dir.mkdir(parents=True, exist_ok=False)
    target = backup_dir / "dp_actor.py.bak"
    shutil.copy2(path, target)
    manifest = {
        "marker": MARKER,
        "file": str(path),
        "backup": str(target),
        "before_sha256": sha256(path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def checks(path: Path) -> dict[str, bool]:
    text = path.read_text(encoding="utf-8")
    return {
        "context_import": "from contextlib import nullcontext" in text,
        "mode_switch": 'VERL_GRAD_SYNC_MODE", "per_micro_batch"' in text,
        "indexed_micro_batches": "for micro_batch_idx, micro_batch in enumerate(micro_batches):" in text,
        "no_sync_context": "self.actor_module.no_sync()" in text,
    }


def apply_patch(verl_root: Path, backup_dir: Path) -> None:
    path = target_file(verl_root)
    current = checks(path)
    if all(current.values()):
        print("Gradient-sync memory-case patch already applied")
        return
    backup(path, backup_dir)

    if not current["context_import"]:
        replace_once(path, "import logging\nimport os\n", "from contextlib import nullcontext\n\nimport logging\nimport os\n")

    if not current["mode_switch"]:
        anchor = "        self.actor_optimizer = actor_optimizer\n        role = \"Ref\" if actor_optimizer is None else \"Actor\"\n"
        addition = (
            anchor
            + "        self.gradient_sync_mode = os.getenv(\"VERL_GRAD_SYNC_MODE\", \"per_micro_batch\")\n"
            + "        if self.gradient_sync_mode not in {\"per_micro_batch\", \"deferred\"}:\n"
            + "            raise ValueError(f\"Unsupported VERL_GRAD_SYNC_MODE={self.gradient_sync_mode}\")\n"
        )
        replace_once(path, anchor, addition)

    if not current["indexed_micro_batches"]:
        replace_once(
            path,
            "                for micro_batch in micro_batches:\n",
            "                for micro_batch_idx, micro_batch in enumerate(micro_batches):\n",
        )

    if not current["no_sync_context"]:
        anchor = (
            "                    if self.scaler is not None:\n"
            "                        self.scaler.scale(loss).backward()\n"
            "                    else:\n"
            "                        loss.backward()\n"
        )
        replacement = (
            "                    defer_sync = (\n"
            "                        self.gradient_sync_mode == \"deferred\"\n"
            "                        and micro_batch_idx + 1 < len(micro_batches)\n"
            "                        and hasattr(self.actor_module, \"no_sync\")\n"
            "                    )\n"
            "                    sync_context = self.actor_module.no_sync() if defer_sync else nullcontext()\n"
            "                    with sync_context:\n"
            "                        if self.scaler is not None:\n"
            "                            self.scaler.scale(loss).backward()\n"
            "                        else:\n"
            "                            loss.backward()\n"
        )
        replace_once(path, anchor, replacement)

    manifest_path = backup_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["after_sha256"] = sha256(path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Applied {MARKER}; mode selected by VERL_GRAD_SYNC_MODE; backup: {backup_dir}")


def restore(backup_dir: Path) -> None:
    manifest = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))
    shutil.copy2(manifest["backup"], manifest["file"])
    print(f"restored: {manifest['file']}")


def check(verl_root: Path) -> None:
    result = checks(target_file(verl_root))
    print(json.dumps(result, indent=2))
    if not all(result.values()):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("apply", "check", "restore"))
    parser.add_argument("--verl-root", type=Path, default=Path("/workspace/verl"))
    parser.add_argument("--backup-dir", type=Path, default=Path("/tmp/verl_gradient_sync_case_backup"))
    args = parser.parse_args()
    if args.command == "apply":
        apply_patch(args.verl_root.resolve(), args.backup_dir.resolve())
    elif args.command == "check":
        check(args.verl_root.resolve())
    else:
        restore(args.backup_dir.resolve())


if __name__ == "__main__":
    main()
