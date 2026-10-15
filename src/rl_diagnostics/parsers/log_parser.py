"""REQ-01: Training log and dynamic parameter parser.

Parses verl training logs to extract:
- Hydra command-line parameters (experiment config)
- Step metrics (training/validation/checkpoint)
- Parse quality report

Does NOT execute any script. Pure static parsing.
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

from ..schemas.data_models import (
    ExperimentRecord,
    ParseReport,
    SCHEMA_VERSION,
    StepMetric,
)

# ── ANSI escape removal ──────────────────────────────────────────────
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Ray process prefix like (pid=521627) or (TaskRunner pid=521627)
_RAY_PREFIX_RE = re.compile(r"^\s*\([^)]*\)\s*")
# Timestamp prefix like [2026-10-09 11:48:43,680]
_TIMESTAMP_RE = re.compile(r"^\s*\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,\d]*\]\s*")

# ── Hydra parameter patterns ─────────────────────────────────────────
# Matches key=value, +key=value, ++key=value
_PARAM_RE = re.compile(
    r"(?<!\S)(\+{0,2})([a-zA-Z_][\w.]*)=([^\s]+)"
)

# ── Metric line patterns ─────────────────────────────────────────────
# "step:1 - actor/entropy:0.733 - ..."
_STEP_PREFIX_RE = re.compile(r"^step:(\d+)\s*-?\s*")
# "key:value" within a metric line
_KV_RE = re.compile(r"([a-zA-Z_][\w/.]*):(-?\d+\.?\d*(?:[eE][+-]?\d+)?|-?\d+|nan|inf|-inf|True|False)")

# ── Validation metrics pattern ───────────────────────────────────────
# "Initial validation metrics: {'key': val, ...}"
_VAL_METRICS_RE = re.compile(r"validation metrics:\s*(\{.*\})")

# ── Known metric name groups ─────────────────────────────────────────
_KNOWN_METRIC_NAMES = {
    "training/global_step", "training/epoch",
    "actor/entropy", "actor/ppo_kl", "actor/kl_loss",
    "actor/pg_clipfrac", "actor/grad_norm", "actor/pg_loss",
    "actor/pg_clipfrac_lower", "actor/kl_coef", "actor/lr",
    "response_length/mean", "response_length/max", "response_length/min",
    "response_length/clip_ratio",
    "perf/throughput", "perf/total_num_tokens", "perf/time_per_step",
    "critic/score/mean", "critic/score/max", "critic/score/min",
    "critic/rewards/mean", "critic/rewards/max", "critic/rewards/min",
    "critic/advantages/mean", "critic/advantages/max", "critic/advantages/min",
}


def _clean_line(line: str) -> str:
    """Remove ANSI codes, Ray prefix, timestamp prefix."""
    line = _ANSI_RE.sub("", line)
    line = _RAY_PREFIX_RE.sub("", line)
    line = _TIMESTAMP_RE.sub("", line)
    return line.strip()


def _parse_value(val_str: str) -> Any:
    """Parse a Hydra parameter value into Python native type."""
    val_str = val_str.strip().strip("'\"")
    # Boolean
    if val_str in ("True", "true"):
        return True
    if val_str in ("False", "false"):
        return False
    if val_str in ("None", "null"):
        return None
    # Try int
    try:
        return int(val_str)
    except ValueError:
        pass
    # Try float
    try:
        return float(val_str)
    except ValueError:
        pass
    # List?
    if val_str.startswith("[") and val_str.endswith("]"):
        inner = val_str[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip().strip("'\"") for p in inner.split(",")]
        result = []
        for p in parts:
            result.append(_parse_value(p))
        return result
    return val_str


def _parse_metric_value(val_str: str) -> Optional[float]:
    """Parse a metric value string to float. Returns None if not parseable."""
    val_str = val_str.strip()
    if val_str in ("nan", "NaN"):
        return float("nan")
    if val_str in ("inf", "Inf"):
        return float("inf")
    if val_str in ("-inf", "-Inf"):
        return float("-inf")
    if val_str in ("True", "False"):
        return None  # Boolean metrics not numeric
    try:
        return float(val_str)
    except ValueError:
        return None


def _extract_command_line(log_text: str) -> Optional[str]:
    """Find the python -m verl.trainer.main_ppo command line in the log."""
    for line in log_text.splitlines():
        cleaned = _clean_line(line)
        if "python" in cleaned and "verl.trainer.main_ppo" in cleaned:
            return cleaned
    return None


def parse_parameters(command_line: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse Hydra key=value parameters from a command line.
    
    Returns (parameters_dict, conflicts_list).
    Last value wins for duplicate keys; conflicts are recorded.
    """
    params: dict[str, Any] = {}
    conflicts: list[dict[str, Any]] = []
    seen: dict[str, list[Any]] = {}
    
    for match in _PARAM_RE.finditer(command_line):
        prefix, key, val = match.group(1), match.group(2), match.group(3)
        parsed_val = _parse_value(val)
        
        if key in params:
            conflicts.append({
                "key": key,
                "previous_value": params[key],
                "new_value": parsed_val,
                "resolution": "last_wins",
            })
            seen.setdefault(key, []).append(params[key])
        
        params[key] = parsed_val
        seen.setdefault(key, []).append(parsed_val)
    
    return params, conflicts


def _build_experiment_record(
    params: dict[str, Any],
    conflicts: list[dict[str, Any]],
    log_file: str,
) -> ExperimentRecord:
    """Build ExperimentRecord from parsed parameters."""
    import hashlib
    
    exp_name = params.get("trainer.experiment_name", "")
    proj_name = params.get("trainer.project_name", "")
    exp_id = hashlib.sha256(
        f"{proj_name}:{exp_name}:{log_file}".encode()
    ).hexdigest()[:16]
    
    def _get(key, default=None):
        return params.get(key, default)
    
    return ExperimentRecord(
        experiment_id=exp_id,
        experiment_name=exp_name,
        model=_get("actor_rollout_ref.model.path", ""),
        algorithm=_get("algorithm.adv_estimator", ""),
        dataset=_get("data.train_files", ""),
        rollout_n=_get("actor_rollout_ref.rollout.n"),
        train_batch_size=_get("data.train_batch_size"),
        learning_rate=_get("actor_rollout_ref.actor.optim.lr"),
        total_epochs=_get("trainer.total_epochs"),
        total_training_steps=_get("trainer.total_training_steps"),
        raw_parameters=params,
        config_sources={"primary": os.path.basename(log_file)},
        config_conflicts=conflicts,
    )


def _parse_metric_line(
    line: str,
    experiment_id: str,
    source_file: str,
    line_no: int,
    epoch_map: dict[int, int],
) -> list[StepMetric]:
    """Parse a single metric line into StepMetric objects."""
    cleaned = _clean_line(line)
    
    # Must start with "step:N"
    step_match = _STEP_PREFIX_RE.match(cleaned)
    if not step_match:
        return []
    
    step = int(step_match.group(1))
    rest = cleaned[step_match.end():]
    epoch = epoch_map.get(step)
    
    metrics: list[StepMetric] = []
    
    for kv_match in _KV_RE.finditer(rest):
        key = kv_match.group(1)
        val_str = kv_match.group(2)
        val = _parse_metric_value(val_str)
        if val is None:
            continue
        
        # Determine phase
        if key.startswith("val") or key.startswith("val-"):
            phase = "validation"
        elif "checkpoint" in key.lower() or "save" in key.lower():
            phase = "checkpoint"
        elif key in ("training/global_step", "training/epoch"):
            phase = "train"
        else:
            phase = "train"
        
        # Update epoch map
        if key == "training/epoch" and epoch is None:
            epoch = int(val)
            epoch_map[step] = epoch
        
        metrics.append(StepMetric(
            experiment_id=experiment_id,
            step=step,
            epoch=epoch,
            phase=phase,
            metric_name=key,
            metric_value=val,
            source_file=source_file,
            source_line=line_no,
        ))
    
    return metrics


def _parse_validation_line(
    line: str,
    experiment_id: str,
    source_file: str,
    line_no: int,
    step: int,
) -> list[StepMetric]:
    """Parse a validation metrics dict line."""
    cleaned = _clean_line(line)
    match = _VAL_METRICS_RE.search(cleaned)
    if not match:
        return []
    
    dict_str = match.group(1)
    try:
        val_dict = eval(dict_str, {"__builtins__": {}}, {})  # noqa: S307 - controlled input
    except Exception:
        return []
    
    metrics: list[StepMetric] = []
    for key, val in val_dict.items():
        if isinstance(val, (int, float)):
            metrics.append(StepMetric(
                experiment_id=experiment_id,
                step=step,
                epoch=None,
                phase="validation",
                metric_name=str(key),
                metric_value=float(val),
                source_file=source_file,
                source_line=line_no,
            ))
    
    return metrics


def parse_log(
    log_path: str,
    script_path: Optional[str] = None,
    experiment_id_override: Optional[str] = None,
) -> tuple[ExperimentRecord, list[StepMetric], ParseReport]:
    """Parse a verl training log file.
    
    Args:
        log_path: Path to the training log file (e.g. successLog.txt, train_run.log).
        script_path: Optional path to start.sh for fallback parameter extraction.
        experiment_id_override: Optional explicit experiment ID.
    
    Returns:
        Tuple of (ExperimentRecord, list[StepMetric], ParseReport).
    """
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        log_text = f.read()
    
    total_lines = len(log_text.splitlines())
    report = ParseReport(experiment_id="", total_lines=total_lines)
    
    # ── Extract command line and parameters ──────────────────────────
    command_line = _extract_command_line(log_text)
    params: dict[str, Any] = {}
    conflicts: list[dict[str, Any]] = []
    config_source = "log"
    
    if command_line:
        params, conflicts = parse_parameters(command_line)
        report.parsed_lines += 1
    elif script_path and os.path.isfile(script_path):
        with open(script_path, "r", errors="replace") as f:
            script_text = f.read()
        command_line = None
        for line in script_text.splitlines():
            if "python" in line and "main_ppo" in line:
                command_line = line
                break
        if command_line:
            params, conflicts = parse_parameters(command_line)
            config_source = "script_unverified"
            report.warnings.append({
                "type": "script_fallback",
                "message": "Parameters extracted from start.sh, actual runtime params not verified.",
            })
        else:
            report.missing_required.append("command_line")
    else:
        report.missing_required.append("command_line")
    
    report.config_conflicts = conflicts
    
    # ── Build experiment record ──────────────────────────────────────
    exp = _build_experiment_record(params, conflicts, log_path)
    if experiment_id_override:
        exp.experiment_id = experiment_id_override
    report.experiment_id = exp.experiment_id
    
    # ── Parse metrics ────────────────────────────────────────────────
    all_metrics: list[StepMetric] = []
    epoch_map: dict[int, int] = {}
    last_step = 0
    failed_count = 0
    
    for line_no, line in enumerate(log_text.splitlines(), 1):
        cleaned = _clean_line(line)
        if not cleaned:
            continue
        
        # Metric line: starts with "step:N"
        if _STEP_PREFIX_RE.match(cleaned):
            step_metrics = _parse_metric_line(
                line, exp.experiment_id, log_path, line_no, epoch_map
            )
            if step_metrics:
                all_metrics.extend(step_metrics)
                last_step = step_metrics[0].step
                report.parsed_lines += 1
            else:
                failed_count += 1
                report.errors.append({
                    "line": line_no,
                    "type": "metric_parse_failure",
                    "content": cleaned[:200],
                })
        
        # Validation metrics line
        elif "validation metrics:" in cleaned:
            val_metrics = _parse_validation_line(
                line, exp.experiment_id, log_path, line_no, last_step
            )
            if val_metrics:
                all_metrics.extend(val_metrics)
                report.parsed_lines += 1
    
    report.failed_lines = failed_count
    
    # ── Identify unknown metric names ────────────────────────────────
    found_names = set()
    for m in all_metrics:
        found_names.add(m.metric_name)
    for name in sorted(found_names - _KNOWN_METRIC_NAMES):
        report.unknown_fields.append(name)
    
    return exp, all_metrics, report
