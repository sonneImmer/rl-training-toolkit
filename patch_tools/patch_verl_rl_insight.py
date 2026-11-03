#!/usr/bin/env python3
"""Add reversible RL-Insight metrics and phase traces to the fixed VERL image."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


MARKER = "zkx-verl-rl-insight-v1"


BRIDGE = '''"""Compatibility bridge between the fixed VERL image and RL-Insight."""

from __future__ import annotations

import functools
import os
import re
from contextlib import contextmanager, nullcontext


_INITIALIZED = False
_STEP = 0


def enabled():
    return os.getenv("RL_INSIGHT_ENABLE", "0").lower() in {"1", "true", "yes", "on"}


def _safe_name(name):
    return re.sub(r"[^a-zA-Z0-9_:]", "_", str(name).replace("/", "_"))


def ensure_initialized():
    global _INITIALIZED
    if _INITIALIZED or not enabled():
        return _INITIALIZED
    try:
        import rl_insight as insight

        insight.init(
            project=os.getenv("RL_INSIGHT_PROJECT", "verl"),
            experiment_name=os.getenv("RL_INSIGHT_EXPERIMENT", "unknown"),
        )
        _INITIALIZED = True
    except Exception as exc:
        print(f"[rl-insight-bridge] initialization failed: {exc}")
    return _INITIALIZED


def set_step(step):
    global _STEP
    _STEP = int(step)


def _labels(labels):
    result = {"global_step": _STEP, "sync_mode": os.getenv("VERL_GRAD_SYNC_MODE", "unknown")}
    result.update({key: value for key, value in labels.items() if value is not None})
    return result


@contextmanager
def phase(name, lane="trainer_0", **labels):
    if not ensure_initialized():
        yield
        return
    import rl_insight as insight

    with insight.trace_state(
        _safe_name(name),
        state_lane_id=str(lane),
        **_labels(labels),
    ):
        yield


@contextmanager
def traced_timer(timer, name, timing_raw, *args, **kwargs):
    lane = "trainer_step" if name == "step" else "trainer_0"
    with phase(name, lane=lane, component="trainer"):
        with timer(name, timing_raw, *args, **kwargs):
            yield


def worker_phase(name):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(self, *args, **kwargs):
            payload = args[0] if args else kwargs.get("data")
            if payload is None:
                payload = kwargs.get("prompts")
            payload_meta = getattr(payload, "meta_info", {}) or {}
            if "global_steps" in payload_meta:
                set_step(payload_meta["global_steps"])
            rank = getattr(self, "rank", None)
            if rank is None:
                try:
                    import torch

                    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                except Exception:
                    rank = 0
            lane = f"npu_rank_{rank}"
            with phase(name, lane=lane, component="worker", rank=int(rank)):
                return function(self, *args, **kwargs)

        return wrapped

    return decorate


class RLInsightLogger:
    def __init__(self):
        ensure_initialized()

    def log(self, data, step):
        if not ensure_initialized():
            return
        set_step(step)
        import rl_insight as insight

        for key, value in data.items():
            if isinstance(value, bool):
                value = int(value)
            if not isinstance(value, (int, float)):
                continue
            name = _safe_name(key)
            metric_labels = _labels({"source_key": str(key)})
            insight.metric_gauge(name, float(value), **metric_labels)
            insight.metric_log(name, float(value), step=int(step), **metric_labels)

    def finish(self):
        global _INITIALIZED
        if not _INITIALIZED:
            return
        try:
            import rl_insight as insight

            insight.finish()
        finally:
            _INITIALIZED = False
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one patch anchor, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def files(verl_root: Path) -> dict[str, Path]:
    return {
        "bridge": verl_root / "verl/utils/rl_insight_bridge.py",
        "tracking": verl_root / "verl/utils/tracking.py",
        "trainer": verl_root / "verl/trainer/ppo/ray_trainer.py",
        "workers": verl_root / "verl/workers/fsdp_workers.py",
    }


def backup(targets: dict[str, Path], backup_dir: Path) -> None:
    manifest_path = backup_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("marker") != MARKER:
            raise RuntimeError(f"{backup_dir}: incompatible backup manifest")
        return
    backup_dir.mkdir(parents=True, exist_ok=False)
    entries = {}
    for name in ("tracking", "trainer", "workers"):
        source = targets[name]
        destination = backup_dir / f"{name}.py.bak"
        shutil.copy2(source, destination)
        entries[name] = {
            "file": str(source),
            "backup": str(destination),
            "before_sha256": sha256(source),
        }
    manifest_path.write_text(
        json.dumps({"marker": MARKER, "entries": entries}, indent=2) + "\n",
        encoding="utf-8",
    )


def checks(targets: dict[str, Path]) -> dict[str, bool]:
    tracking = targets["tracking"].read_text(encoding="utf-8")
    trainer = targets["trainer"].read_text(encoding="utf-8")
    workers = targets["workers"].read_text(encoding="utf-8")
    return {
        "bridge": targets["bridge"].is_file() and MARKER in targets["bridge"].read_text(encoding="utf-8"),
        "tracking_logger": 'self.logger["rl_insight"] = RLInsightLogger()' in tracking,
        "trainer_timer": "with traced_timer(marked_timer," in trainer,
        "trainer_step": "set_rl_insight_step(self.global_steps)" in trainer,
        "worker_decorators": workers.count("@worker_phase(") >= 4,
    }


def apply_patch(verl_root: Path, backup_dir: Path) -> None:
    targets = files(verl_root)
    if all(checks(targets).values()):
        print("RL-Insight VERL patch already applied")
        return
    backup(targets, backup_dir)
    targets["bridge"].write_text(f'# {MARKER}\n{BRIDGE}', encoding="utf-8")

    tracking = targets["tracking"]
    if 'self.logger["rl_insight"] = RLInsightLogger()' not in tracking.read_text(encoding="utf-8"):
        replace_once(
            tracking,
            "        self.logger = {}\n",
            "        self.logger = {}\n"
            "        import os as _rl_insight_os\n"
            "        if _rl_insight_os.getenv(\"RL_INSIGHT_ENABLE\", \"0\").lower() in {\"1\", \"true\", \"yes\", \"on\"}:\n"
            "            from verl.utils.rl_insight_bridge import RLInsightLogger\n"
            "            self.logger[\"rl_insight\"] = RLInsightLogger()\n",
        )
        replace_once(
            tracking,
            '        if "file" in self.logger:\n            self.logger["file"].finish()\n',
            '        if "file" in self.logger:\n            self.logger["file"].finish()\n'
            '        if "rl_insight" in self.logger:\n            self.logger["rl_insight"].finish()\n',
        )

    trainer = targets["trainer"]
    trainer_text = trainer.read_text(encoding="utf-8")
    if "from verl.utils.rl_insight_bridge import" not in trainer_text:
        replace_once(
            trainer,
            "from verl.utils.py_functional import rename_dict\n",
            "from verl.utils.py_functional import rename_dict\n"
            "from verl.utils.rl_insight_bridge import set_step as set_rl_insight_step, traced_timer\n",
        )
    trainer_text = trainer.read_text(encoding="utf-8")
    trainer_text = trainer_text.replace("with marked_timer(", "with traced_timer(marked_timer,")
    trainer.write_text(trainer_text, encoding="utf-8")
    if "set_rl_insight_step(self.global_steps)" not in trainer.read_text(encoding="utf-8"):
        replace_once(
            trainer,
            "                metrics = {}\n                timing_raw = {}\n",
            "                metrics = {}\n                timing_raw = {}\n                set_rl_insight_step(self.global_steps)\n",
        )

    workers = targets["workers"]
    if "from verl.utils.rl_insight_bridge import worker_phase" not in workers.read_text(encoding="utf-8"):
        replace_once(
            workers,
            "from verl.utils.py_functional import convert_to_regular_types\n",
            "from verl.utils.py_functional import convert_to_regular_types\n"
            "from verl.utils.rl_insight_bridge import worker_phase\n",
        )
    worker_text = workers.read_text(encoding="utf-8")
    decorators = {
        "    @DistProfiler.annotate(color=\"red\", role=\"actor_update\")\n    def update_actor":
            "    @DistProfiler.annotate(color=\"red\", role=\"actor_update\")\n    @worker_phase(\"actor_update\")\n    def update_actor",
        "    @DistProfiler.annotate(color=\"red\", role=\"rollout_generate\")\n    def generate_sequences":
            "    @DistProfiler.annotate(color=\"red\", role=\"rollout_generate\")\n    @worker_phase(\"rollout_generate\")\n    def generate_sequences",
        "    @DistProfiler.annotate(color=\"blue\", role=\"actor_compute_log_prob\")\n    def compute_log_prob":
            "    @DistProfiler.annotate(color=\"blue\", role=\"actor_compute_log_prob\")\n    @worker_phase(\"actor_compute_log_prob\")\n    def compute_log_prob",
        "    @DistProfiler.annotate(color=\"olive\", role=\"ref_compute_log_prob\")\n    def compute_ref_log_prob":
            "    @DistProfiler.annotate(color=\"olive\", role=\"ref_compute_log_prob\")\n    @worker_phase(\"ref_compute_log_prob\")\n    def compute_ref_log_prob",
    }
    for old, new in decorators.items():
        if new not in worker_text:
            if worker_text.count(old) != 1:
                raise RuntimeError(f"{workers}: expected one worker decorator anchor for {old!r}")
            worker_text = worker_text.replace(old, new)
    workers.write_text(worker_text, encoding="utf-8")

    manifest_path = backup_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("tracking", "trainer", "workers"):
        manifest["entries"][name]["after_sha256"] = sha256(targets[name])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Applied {MARKER}; backup: {backup_dir}")


def restore(verl_root: Path, backup_dir: Path) -> None:
    targets = files(verl_root)
    manifest = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["entries"].values():
        shutil.copy2(entry["backup"], entry["file"])
    targets["bridge"].unlink(missing_ok=True)
    print(f"Restored {MARKER}")


def check(verl_root: Path) -> None:
    result = checks(files(verl_root))
    print(json.dumps(result, indent=2))
    if not all(result.values()):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("apply", "check", "restore"))
    parser.add_argument("--verl-root", type=Path, default=Path("/workspace/verl"))
    parser.add_argument("--backup-dir", type=Path, default=Path("/tmp/verl_rl_insight_backup"))
    args = parser.parse_args()
    if args.command == "apply":
        apply_patch(args.verl_root.resolve(), args.backup_dir.resolve())
    elif args.command == "check":
        check(args.verl_root.resolve())
    else:
        restore(args.verl_root.resolve(), args.backup_dir.resolve())


if __name__ == "__main__":
    main()
