"""Persistent experiment objects built from training run directories."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .analyzers.group_signal import aggregate_step, diagnose_group_signal
from .analyzers.rollout_quality import assess_rollout_quality
from .analyzers.stability_analyzer import run_stability_diagnosis
from .config.stability_config import StabilityConfig
from .experiment_rules import RULE_VERSION, RuleConfig, evaluate_experiment_rules, rule_config_dict
from .live_monitor import build_snapshot, latest_rollouts
from .parsers.log_parser import parse_log
from .parsers.rollout_parser import parse_rollout_dir


ERROR_MARKERS = re.compile(r"Traceback|RuntimeError|OutOfMemory|ERROR|Exception")
COMPLETE_MARKERS = re.compile(r"Training completed|training finished|Finished training", re.IGNORECASE)
PROGRESS_MARKER = re.compile(r"Training Progress:.*?(\d+)/(\d+)")


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _atomic_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


class ExperimentStore:
    def __init__(
        self,
        runs_root: str | Path,
        state_dir: str | Path,
        stale_seconds: int = 300,
        rule_config: Optional[RuleConfig] = None,
    ) -> None:
        self.runs_root = Path(runs_root).resolve()
        self.state_dir = Path(state_dir).resolve()
        self.stale_seconds = stale_seconds
        self.rule_config = rule_config or RuleConfig()
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def discover_run_dirs(self) -> list[Path]:
        if not self.runs_root.is_dir():
            return []
        return sorted({path.parent for path in self.runs_root.rglob("train.log")})

    def experiment_id(self, run_dir: Path) -> str:
        digest = hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:12]
        name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", run_dir.name).strip("-") or "experiment"
        return f"{name}-{digest}"

    def _object_dir(self, experiment_id: str) -> Path:
        return self.state_dir / "experiments" / experiment_id

    def _read_previous(self, experiment_id: str) -> Optional[dict]:
        path = self._object_dir(experiment_id) / "snapshot.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _activity_mtime(self, run_dir: Path) -> float:
        candidates = [run_dir / "train.log"]
        rollout_dir = run_dir / "rollouts"
        if rollout_dir.is_dir():
            candidates.extend(rollout_dir.glob("*.jsonl"))
        mtimes = [path.stat().st_mtime for path in candidates if path.exists()]
        return max(mtimes) if mtimes else run_dir.stat().st_mtime

    def _read_log_tail(self, log_path: Path) -> str:
        if not log_path.is_file():
            return ""
        with log_path.open("rb") as handle:
            handle.seek(max(0, log_path.stat().st_size - 131072))
            return handle.read().decode(errors="replace")

    def _infer_total_steps(self, tail: str, configured_total: Optional[int]) -> Optional[int]:
        if configured_total:
            return configured_total
        matches = PROGRESS_MARKER.findall(tail)
        return int(matches[-1][1]) if matches else None

    def _infer_status(
        self,
        run_dir: Path,
        latest_step: Optional[int],
        total_steps: Optional[int],
        tail: str,
    ) -> str:
        has_error = bool(ERROR_MARKERS.search(tail[-16000:]))
        completed = bool(COMPLETE_MARKERS.search(tail))
        completed = completed or bool(total_steps and latest_step is not None and latest_step >= total_steps)
        if completed:
            return "completed_with_errors" if has_error else "completed"
        if has_error:
            return "failed"
        if time.time() - self._activity_mtime(run_dir) <= self.stale_seconds:
            return "running"
        return "stopped"

    def _build_diagnostics(self, experiment_id: str, rollout_dir: Path, metrics: list) -> dict:
        if not rollout_dir.is_dir():
            return {
                "quality_by_step": [],
                "group_signal_by_step": [],
                "group_details": [],
                "parser_warnings": [],
                "stability": self._stability_payload(metrics, []),
            }

        rollouts, groups, warnings = parse_rollout_dir(str(rollout_dir), experiment_id)
        quality_results = [assess_rollout_quality(rollout) for rollout in rollouts]
        rollout_lookup = {
            (rollout.step, rollout.group_id, rollout.rollout_id): rollout
            for rollout in rollouts
        }
        group_results = []
        for group in groups:
            labels = []
            has_label = False
            for rollout_id in group.rollout_ids:
                rollout = rollout_lookup.get((group.step, group.group_id, rollout_id))
                label = rollout.verifier_correct if rollout else None
                labels.append(label)
                has_label = has_label or label is not None
            group_results.append(diagnose_group_signal(
                group,
                quality_results,
                correctness_labels=labels if has_label else None,
            ))

        steps = sorted({rollout.step for rollout in rollouts} | {group.step for group in groups})
        step_aggs = [aggregate_step(step, None, group_results, quality_results) for step in steps]
        quality_by_step = []
        for step in steps:
            counts = Counter(result.status for result in quality_results if result.step == step)
            total = sum(counts.values())
            quality_by_step.append({
                "step": step,
                "total": total,
                "usable_ratio": counts["usable"] / total if total else None,
                "counts": dict(sorted(counts.items())),
            })

        return {
            "quality_by_step": quality_by_step,
            "group_signal_by_step": [item.to_dict() for item in step_aggs],
            "group_details": [item.to_dict() for item in group_results],
            "parser_warnings": warnings,
            "stability": self._stability_payload(metrics, step_aggs),
        }

    @staticmethod
    def _stability_payload(metrics: list, step_aggs: list) -> dict:
        config = StabilityConfig()
        report = run_stability_diagnosis(metrics, step_aggs, config)
        alerts_by_rule = {alert.rule_name: alert.to_dict() for alert in report.alerts}
        metric_names = set(report.metrics_available)
        evaluations = []
        for rule_name, rule_config in config.get_rule_configs().items():
            alert = alerts_by_rule.get(rule_name)
            if alert:
                evaluations.append({"rule_name": rule_name, "status": "triggered", **alert})
                continue
            required = set(rule_config.extra.get("metric_names", []))
            if rule_name == "reward_signal_degradation":
                available = bool(step_aggs)
                sample_count = len(step_aggs)
                minimum = rule_config.baseline_window + rule_config.window_size
            else:
                matching = [metric for metric in metrics if metric.metric_name in required]
                available = bool(required & metric_names)
                sample_count = max(
                    (sum(1 for metric in matching if metric.metric_name == name) for name in required),
                    default=0,
                )
                minimum = rule_config.baseline_window + rule_config.window_size
                if rule_name == "gradient_spike":
                    minimum = rule_config.baseline_window + 1
                elif rule_name == "clip_saturation":
                    minimum = rule_config.window_size * rule_config.min_windows
                elif rule_name == "kl_drift":
                    minimum = rule_config.baseline_window + rule_config.window_size * rule_config.min_windows
            if not rule_config.enabled:
                status, reason = "not_evaluated", "rule_disabled"
            elif not available:
                status, reason = "not_evaluated", "required_metric_missing"
            elif sample_count < minimum:
                status, reason = "not_evaluated", f"insufficient_samples:{sample_count}/{minimum}"
            else:
                status, reason = "not_triggered", "threshold_not_reached"
            evaluations.append({
                "rule_name": rule_name,
                "status": status,
                "reason": reason,
                "severity": "info",
                "threshold": rule_config.threshold,
                "window_size": rule_config.window_size,
                "baseline_window": rule_config.baseline_window,
                "config_version": config.config_version,
            })
        payload = report.to_dict()
        payload["evaluations"] = evaluations
        payload["rule_config"] = config.to_dict()
        return payload

    @staticmethod
    def _merge_diagnostics(steps: list[dict], diagnostics: dict) -> None:
        group_by_step = {item["step"]: item for item in diagnostics["group_signal_by_step"]}
        quality_by_step = {item["step"]: item for item in diagnostics["quality_by_step"]}
        for row in steps:
            row.update(group_by_step.get(row["step"], {}))
            quality = quality_by_step.get(row["step"])
            row["rollout_quality_counts"] = quality["counts"] if quality else {}
            for status, count in row["rollout_quality_counts"].items():
                row[f"quality_{status}"] = count

    def refresh_run(self, run_dir: str | Path) -> dict:
        run_dir = Path(run_dir).resolve()
        experiment_id = self.experiment_id(run_dir)
        log_path = run_dir / "train.log"
        rollout_dir = run_dir / "rollouts"
        exp, metrics, parse_report = parse_log(str(log_path), experiment_id_override=experiment_id)
        live = build_snapshot(log_path, rollout_dir)
        diagnostics = self._build_diagnostics(experiment_id, rollout_dir, metrics)
        self._merge_diagnostics(live["steps"], diagnostics)
        latest_step = live.get("latest_step")
        log_tail = self._read_log_tail(log_path)
        total_steps = self._infer_total_steps(log_tail, exp.total_training_steps)
        status = self._infer_status(run_dir, latest_step, total_steps, log_tail)
        activity_mtime = self._activity_mtime(run_dir)
        started_mtime = min(log_path.stat().st_mtime, run_dir.stat().st_mtime)
        labels = evaluate_experiment_rules(live["steps"], status, self.rule_config)
        latest = live["steps"][-1] if live["steps"] else {}
        progress_ratio = None
        if latest_step is not None and total_steps:
            progress_ratio = min(latest_step / total_steps, 1.0)

        snapshot = {
            "schema_version": "1.0.0",
            "experiment_id": experiment_id,
            "name": exp.experiment_name or run_dir.parent.name,
            "status": status,
            "run_dir": str(run_dir),
            "started_at": _iso(started_mtime),
            "updated_at": _iso(activity_mtime),
            "finished_at": _iso(activity_mtime) if status in ("completed", "completed_with_errors", "failed", "stopped") else None,
            "config": {
                "model": exp.model,
                "algorithm": exp.algorithm,
                "dataset": exp.dataset,
                "rollout_n": exp.rollout_n,
                "train_batch_size": exp.train_batch_size,
                "learning_rate": exp.learning_rate,
                "total_epochs": exp.total_epochs,
                "total_training_steps": total_steps,
            },
            "config_parameters": exp.raw_parameters,
            "config_sources": exp.config_sources,
            "config_conflicts": exp.config_conflicts,
            "progress": {
                "current_step": latest_step,
                "total_steps": total_steps,
                "ratio": progress_ratio,
            },
            "latest_metrics": latest,
            "steps": live["steps"],
            "labels": labels,
            "rule_version": RULE_VERSION,
            "rule_config": rule_config_dict(self.rule_config),
            "parse_quality": {
                "failed_lines": parse_report.failed_lines,
                "missing_required": parse_report.missing_required,
            },
            "diagnostics": diagnostics,
            "artifacts": {
                "train_log": str(log_path),
                "rollout_dir": str(rollout_dir),
                "diagnostics_dir": str(run_dir / "diagnostics"),
                "report_materials_dir": str(run_dir / "report_materials"),
            },
        }

        previous = self._read_previous(experiment_id)
        object_dir = self._object_dir(experiment_id)
        _atomic_json(object_dir / "manifest.json", {
            key: snapshot[key] for key in (
                "schema_version", "experiment_id", "name", "status", "run_dir",
                "started_at", "updated_at", "finished_at", "config", "progress",
                "rule_version", "rule_config", "artifacts",
            )
        })
        _atomic_json(object_dir / "snapshot.json", snapshot)
        if status in ("completed", "completed_with_errors", "failed", "stopped"):
            _atomic_json(object_dir / "final_snapshot.json", snapshot)
        self._append_label_events(object_dir, previous, snapshot)
        return snapshot

    def _append_label_events(self, object_dir: Path, previous: Optional[dict], snapshot: dict) -> None:
        before = {label["name"] for label in (previous or {}).get("labels", [])}
        after = {label["name"] for label in snapshot.get("labels", [])}
        events = []
        for name in sorted(after - before):
            events.append({"type": "label_activated", "label": name})
        for name in sorted(before - after):
            events.append({"type": "label_resolved", "label": name})
        if not events:
            return
        path = object_dir / "events.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for event in events:
                event.update({
                    "experiment_id": snapshot["experiment_id"],
                    "step": snapshot["progress"]["current_step"],
                    "time": datetime.now(timezone.utc).isoformat(),
                    "rule_version": RULE_VERSION,
                })
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def refresh_all(self) -> list[dict]:
        snapshots = []
        for run_dir in self.discover_run_dirs():
            try:
                snapshots.append(self.refresh_run(run_dir))
            except (OSError, ValueError) as error:
                snapshots.append({
                    "experiment_id": self.experiment_id(run_dir),
                    "name": run_dir.name,
                    "status": "unreadable",
                    "run_dir": str(run_dir),
                    "error": str(error),
                    "labels": [],
                    "steps": [],
                    "latest_metrics": {},
                    "progress": {"current_step": None, "total_steps": None, "ratio": None},
                    "config": {},
                })
        snapshots.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        _atomic_json(self.state_dir / "index.json", [self._index_item(item) for item in snapshots])
        return snapshots

    @staticmethod
    def _index_item(item: dict) -> dict:
        latest = item.get("latest_metrics", {})
        return {
            "experiment_id": item.get("experiment_id"),
            "name": item.get("name"),
            "status": item.get("status"),
            "run_dir": item.get("run_dir"),
            "started_at": item.get("started_at"),
            "updated_at": item.get("updated_at"),
            "finished_at": item.get("finished_at"),
            "config": item.get("config", {}),
            "progress": item.get("progress", {}),
            "labels": item.get("labels", []),
            "latest_metrics": latest,
            "rule_version": item.get("rule_version"),
        }

    def list_experiments(self) -> list[dict]:
        path = self.state_dir / "index.json"
        if not path.is_file():
            return self.refresh_all()
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self.refresh_all()

    def get_experiment(self, experiment_id: str) -> Optional[dict]:
        return self._read_previous(experiment_id)

    def compare_experiments(self, experiment_ids: list[str]) -> Optional[dict]:
        experiments = [self.get_experiment(experiment_id) for experiment_id in experiment_ids]
        if any(item is None for item in experiments):
            return None
        items = [item for item in experiments if item is not None]
        parameter_maps = []
        for item in items:
            params = dict(item.get("config_parameters", {}))
            for key, value in item.get("config", {}).items():
                params.setdefault(key, value)
            parameter_maps.append(params)
        parameter_keys = sorted(set().union(*(params.keys() for params in parameter_maps)))
        config_rows = []
        for key in parameter_keys:
            values = [params.get(key) for params in parameter_maps]
            normalized = {json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values}
            config_rows.append({"key": key, "values": values, "changed": len(normalized) > 1})

        metric_keys = (
            "reward_mean", "valid_ratio", "clip_ratio", "entropy", "kl_loss", "grad_norm",
            "rollout_usable_ratio", "effective_group_ratio", "low_information_group_ratio",
            "mean_informative_pair_ratio",
        )
        latest_rows = []
        for key in metric_keys:
            values = [item.get("latest_metrics", {}).get(key) for item in items]
            if any(value is not None for value in values):
                latest_rows.append({
                    "key": key,
                    "values": values,
                    "delta": values[1] - values[0] if len(values) == 2 and all(isinstance(value, (int, float)) for value in values) else None,
                })

        return {
            "experiments": [{
                "experiment_id": item["experiment_id"],
                "name": item.get("name"),
                "status": item.get("status"),
                "config": item.get("config", {}),
            } for item in items],
            "config_rows": config_rows,
            "latest_metric_rows": latest_rows,
            "series": [{
                "experiment_id": item["experiment_id"],
                "name": item.get("name"),
                "steps": item.get("steps", []),
            } for item in items],
        }

    def get_rollouts(self, experiment_id: str, step: Optional[int] = None, limit: int = 12) -> list[dict]:
        snapshot = self.get_experiment(experiment_id)
        if not snapshot:
            return []
        rollout_dir = Path(snapshot["artifacts"]["rollout_dir"])
        if step is None:
            return latest_rollouts(rollout_dir, limit)
        path = rollout_dir / f"{step}.jsonl"
        if not path.is_file():
            return []
        records = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(records) >= limit:
                break
        return records
