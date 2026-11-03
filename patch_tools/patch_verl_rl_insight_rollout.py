#!/usr/bin/env python3
# Backport RL-Insight rollout metrics and replica traces to the fixed VERL image.

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

MARKER = "zkx-verl-rl-insight-rollout-v1"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one patch anchor, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def targets(verl_root: Path) -> dict[str, Path]:
    return {
        "bridge": verl_root / "verl/utils/rl_insight_bridge.py",
        "agent_loop": verl_root / "verl/experimental/agent_loop/agent_loop.py",
        "vllm_server": verl_root / "verl/workers/rollout/vllm_rollout/vllm_async_server.py",
    }


def checks(files: dict[str, Path]) -> dict[str, bool]:
    bridge = files["bridge"].read_text(encoding="utf-8") if files["bridge"].is_file() else ""
    agent_loop = files["agent_loop"].read_text(encoding="utf-8")
    vllm_server = files["vllm_server"].read_text(encoding="utf-8")
    return {
        "bridge_rollout_helpers": "def register_rollout_metrics(" in bridge and "def async_phase(" in bridge,
        "rollout_registration": "register_rollout_metrics(\n            self.server_addresses" in agent_loop,
        "vllm_generate_trace": '@async_phase("vllm_generate"' in vllm_server,
        "vllm_lifecycle_traces": '@async_phase("vllm_wake_up"' in vllm_server and '@async_phase("vllm_sleep"' in vllm_server,
        "served_model_name": "RL_INSIGHT_ENABLE" in vllm_server and "served_model_name" in vllm_server,
    }


def backup(files: dict[str, Path], backup_dir: Path) -> None:
    manifest_path = backup_dir / "manifest.json"
    if manifest_path.exists():
        return
    backup_dir.mkdir(parents=True, exist_ok=False)
    entries = {}
    for name in ("bridge", "agent_loop", "vllm_server"):
        source = files[name]
        destination = backup_dir / f"{name}.py.bak"
        shutil.copy2(source, destination)
        entries[name] = {"file": str(source), "backup": str(destination)}
    manifest_path.write_text(json.dumps({"marker": MARKER, "entries": entries}, indent=2) + "\n", encoding="utf-8")


def apply(verl_root: Path, backup_dir: Path) -> None:
    files = targets(verl_root)
    if all(checks(files).values()):
        print("RL-Insight rollout patch already applied")
        return
    if not files["bridge"].exists():
        raise RuntimeError("Apply patch_verl_rl_insight.py before this rollout patch")
    backup(files, backup_dir)

    bridge = files["bridge"]
    if "def async_phase(" not in bridge.read_text(encoding="utf-8"):
        helper = """

def async_phase(name, lane_getter):
    def decorate(function):
        @functools.wraps(function)
        async def wrapped(self, *args, **kwargs):
            lane = lane_getter(self) if callable(lane_getter) else lane_getter
            with phase(name, lane=lane, component="rollout", replica=getattr(self, "replica_rank", -1)):
                return await function(self, *args, **kwargs)

        return wrapped

    return decorate


def register_rollout_metrics(server_addresses, labels=None):
    if not ensure_initialized() or not server_addresses:
        return
    import rl_insight as insight

    insight.update_prometheus_config(server_addresses, "vllm", labels)


class RLInsightLogger:
"""
        replace_once(bridge, "\n\nclass RLInsightLogger:\n", helper)

    agent_loop = files["agent_loop"]
    if "from verl.utils.rl_insight_bridge import register_rollout_metrics" not in agent_loop.read_text(encoding="utf-8"):
        replace_once(
            agent_loop,
            "from verl.utils.model import compute_position_id_with_mask\n",
            "from verl.utils.model import compute_position_id_with_mask\n"
            "from verl.utils.rl_insight_bridge import register_rollout_metrics\n",
        )
    if "register_rollout_metrics(\n            self.server_addresses" not in agent_loop.read_text(encoding="utf-8"):
        replace_once(
            agent_loop,
            '        print(f"AgentLoopManager: {self.server_addresses}")\n\n',
            '        print(f"AgentLoopManager: {self.server_addresses}")\n'
            '        register_rollout_metrics(\n'
            '            self.server_addresses,\n'
            '            labels=[{"replica": replica.replica_rank} for replica in self.rollout_replicas],\n'
            '        )\n\n',
        )

    vllm_server = files["vllm_server"]
    if "from verl.utils.rl_insight_bridge import async_phase" not in vllm_server.read_text(encoding="utf-8"):
        replace_once(
            vllm_server,
            "from verl.utils.config import omega_conf_to_dataclass\n",
            "from verl.utils.config import omega_conf_to_dataclass\n"
            "from verl.utils.rl_insight_bridge import async_phase\n",
        )
    text = vllm_server.read_text(encoding="utf-8")
    old_prometheus = "        if self.config.prometheus.enable:\n            if self.config.prometheus.served_model_name:\n"
    new_prometheus = (
        "        if self.config.prometheus.enable or os.getenv(\"RL_INSIGHT_ENABLE\", \"0\").lower() in "
        "{\"1\", \"true\", \"yes\", \"on\"}:\n"
        "            if self.config.prometheus.served_model_name:\n"
    )
    if new_prometheus not in text:
        if text.count(old_prometheus) != 1:
            raise RuntimeError(f"{vllm_server}: expected one prometheus anchor")
        text = text.replace(old_prometheus, new_prometheus)
    decorators = {
        "    async def generate(\n": '    @async_phase("vllm_generate", lambda self: f"vllm_replica_{self.replica_rank}")\n    async def generate(\n',
        "    async def wake_up(self):\n        if self.rollout_mode == RolloutMode.HYBRID:\n": '    @async_phase("vllm_wake_up", lambda self: f"vllm_replica_{self.replica_rank}")\n    async def wake_up(self):\n        if self.rollout_mode == RolloutMode.HYBRID:\n',
        "    async def sleep(self):\n        if self.rollout_mode == RolloutMode.HYBRID:\n": '    @async_phase("vllm_sleep", lambda self: f"vllm_replica_{self.replica_rank}")\n    async def sleep(self):\n        if self.rollout_mode == RolloutMode.HYBRID:\n',
    }
    for old, new in decorators.items():
        if new not in text:
            if text.count(old) != 1:
                raise RuntimeError(f"{vllm_server}: expected one decorator anchor for {old!r}")
            text = text.replace(old, new)
    vllm_server.write_text(text, encoding="utf-8")

    result = checks(files)
    if not all(result.values()):
        raise RuntimeError(f"Incomplete rollout patch: {result}")
    print(f"Applied {MARKER}; backup: {backup_dir}")


def restore(verl_root: Path, backup_dir: Path) -> None:
    manifest = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["entries"].values():
        shutil.copy2(entry["backup"], entry["file"])
    print(f"Restored {MARKER}")


def check(verl_root: Path) -> None:
    result = checks(targets(verl_root))
    print(json.dumps(result, indent=2))
    if not all(result.values()):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("apply", "check", "restore"))
    parser.add_argument("--verl-root", type=Path, default=Path("/workspace/verl"))
    parser.add_argument("--backup-dir", type=Path, default=Path("/tmp/verl_rl_insight_rollout_backup"))
    args = parser.parse_args()
    if args.command == "apply":
        apply(args.verl_root.resolve(), args.backup_dir.resolve())
    elif args.command == "check":
        check(args.verl_root.resolve())
    else:
        restore(args.verl_root.resolve(), args.backup_dir.resolve())


if __name__ == "__main__":
    main()
