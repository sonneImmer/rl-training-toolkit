"""Unified data schemas for RL diagnostics.

All modules must use these dataclasses as the canonical data contract.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = "1.0.0"


def _to_jsonable(obj: Any) -> Any:
    """Convert numpy/torch types to JSON-serializable Python natives."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


@dataclass
class ExperimentRecord:
    """REQ-01: experiment configuration extracted from training log."""
    experiment_id: str
    experiment_name: str = ""
    model: str = ""
    algorithm: str = ""
    dataset: str = ""
    rollout_n: Optional[int] = None
    train_batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    total_epochs: Optional[int] = None
    total_training_steps: Optional[int] = None
    raw_parameters: dict[str, Any] = field(default_factory=dict)
    config_sources: dict[str, str] = field(default_factory=dict)
    config_conflicts: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(asdict(self))


@dataclass
class StepMetric:
    """REQ-01: a single metric value at a training step."""
    experiment_id: str
    step: int
    epoch: Optional[int]
    phase: str  # train | validation | checkpoint | unknown
    metric_name: str
    metric_value: float
    source_file: str = ""
    source_line: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(asdict(self))


@dataclass
class ParseReport:
    """REQ-01: parse quality report."""
    experiment_id: str
    total_lines: int = 0
    parsed_lines: int = 0
    failed_lines: int = 0
    unknown_fields: list[str] = field(default_factory=list)
    config_conflicts: list[dict[str, Any]] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(asdict(self))


@dataclass
class RolloutRecord:
    """REQ-02: standardized single rollout."""
    schema_version: str
    experiment_id: str
    step: int
    epoch: Optional[int]
    generation_attempt: Optional[int]
    group_id: str
    rollout_id: int
    prompt_id: Optional[str] = None
    id_type: str = "unknown"  # unknown | original | content_fingerprint
    prompt: Optional[str] = None
    response: Optional[str] = None
    training_reward: Optional[float] = None
    reward_source: str = "training"
    advantage: Optional[float] = None
    advantage_estimator: Optional[str] = None
    reward_components: dict[str, Any] = field(default_factory=dict)
    selected_for_update: Optional[bool] = None
    response_mask_tokens: Optional[int] = None
    policy_signal_tokens: Optional[int] = None
    prediction: Optional[str] = None
    verifier_correct: Optional[bool] = None
    ground_truth: Optional[str] = None
    difficulty: Optional[str] = None
    source_dataset: Optional[str] = None
    record_phase: str = "train"  # train | validation
    source_file: str = ""
    source_line: int = 0
    parse_status: str = "ok"  # ok | schema_error | json_error
    parse_error: Optional[str] = None
    finish_reason: Optional[str] = None
    response_token_length: Optional[int] = None
    max_response_length: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(asdict(self))


@dataclass
class GroupRecord:
    """REQ-02: a group of rollouts sharing the same prompt context."""
    schema_version: str
    experiment_id: str
    step: int
    epoch: Optional[int]
    generation_attempt: Optional[int]
    group_id: str
    prompt_id: Optional[str]
    prompt: Optional[str]
    source_dataset: Optional[str]
    difficulty: Optional[str]
    ground_truth: Optional[str]
    rollout_count: int = 0
    rewards: list[float] = field(default_factory=list)
    advantages: list[Optional[float]] = field(default_factory=list)
    advantage_estimator: Optional[str] = None
    selected_for_update: list[Optional[bool]] = field(default_factory=list)
    responses: list[Optional[str]] = field(default_factory=list)
    rollout_ids: list[int] = field(default_factory=list)
    source_file: str = ""
    group_status: str = "resolved"  # resolved | group_unresolved

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(asdict(self))


def content_fingerprint(text: str) -> str:
    """Generate a deterministic content fingerprint for prompt text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
