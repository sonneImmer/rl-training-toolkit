"""REQ-02: Rollout JSONL parser and Group recovery.

Scans rollout directory for `<step>.jsonl` files, parses each record,
and recovers Group structure.

Supports both old format (input/output/score) and new format
(prompt/response/reward with group_id/rollout_id).
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Optional

from ..schemas.data_models import (
    GroupRecord,
    RolloutRecord,
    SCHEMA_VERSION,
    content_fingerprint,
)

# Valid step filename pattern: integer.jsonl
_STEP_FILE_RE = re.compile(r"^(\d+)\.jsonl$")


def _is_finite_number(val: Any) -> bool:
    """Check if value is a finite number."""
    if val is None:
        return False
    if isinstance(val, bool):
        return False
    if isinstance(val, (int, float)):
        return not (math.isnan(val) or math.isinf(val))
    return False


def _extract_reward(record: dict[str, Any], idx: int = 0) -> Optional[float]:
    """Extract reward value from a record, handling scalar and array forms."""
    for key in ("score", "reward", "training_reward"):
        if key in record:
            val = record[key]
            if isinstance(val, list):
                if idx < len(val):
                    v = val[idx]
                    if _is_finite_number(v):
                        return float(v)
                return None
            if _is_finite_number(val):
                return float(val)
            return None
    return None


def _extract_advantage(record: dict[str, Any], idx: int = 0) -> Optional[float]:
    """Extract a sequence-level advantage when the trainer dumped one."""
    val = record.get("advantage")
    if isinstance(val, list):
        val = val[idx] if idx < len(val) else None
    return float(val) if _is_finite_number(val) else None


def _extract_object_field(record: dict[str, Any], key: str, idx: int = 0) -> dict[str, Any]:
    """Extract a JSON object from scalar or per-response array form."""
    value = record.get(key)
    if isinstance(value, list):
        value = value[idx] if idx < len(value) else None
    return value if isinstance(value, dict) else {}


def _extract_int_field(record: dict[str, Any], key: str, idx: int = 0) -> Optional[int]:
    value = record.get(key)
    if isinstance(value, list):
        value = value[idx] if idx < len(value) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _extract_str_field(record: dict[str, Any], keys: list[str], idx: int = 0) -> Optional[str]:
    """Extract a string field from a record, handling scalar and array forms."""
    for key in keys:
        if key in record:
            val = record[key]
            if isinstance(val, list):
                if idx < len(val) and val[idx] is not None:
                    return str(val[idx])
                return None
            if val is not None:
                return str(val)
    return None


def _extract_bool_field(record: dict[str, Any], keys: list[str], idx: int = 0) -> Optional[bool]:
    """Extract an explicit boolean without guessing from reward values."""
    for key in keys:
        if key not in record:
            continue
        val = record[key]
        if isinstance(val, list):
            val = val[idx] if idx < len(val) else None
        if isinstance(val, bool):
            return val
    return None


def _extract_step(record: dict[str, Any], filename_step: int) -> tuple[int, Optional[str]]:
    """Extract step from record, cross-check with filename."""
    rec_step = record.get("step", None)
    if rec_step is not None and isinstance(rec_step, (int, float)):
        if int(rec_step) != filename_step:
            return filename_step, f"step_mismatch: file={filename_step}, record={int(rec_step)}"
        return int(rec_step), None
    return filename_step, None


def scan_rollout_dir(
    rollout_dir: str,
) -> tuple[list[tuple[int, str]], list[dict[str, Any]]]:
    """Scan a rollout directory for valid `<step>.jsonl` files.
    
    Returns (sorted list of (step, filepath), list of warnings for invalid filenames).
    """
    step_files: list[tuple[int, str]] = []
    warnings: list[dict[str, Any]] = []
    
    if not os.path.isdir(rollout_dir):
        warnings.append({
            "type": "dir_not_found",
            "path": rollout_dir,
        })
        return [], warnings
    
    for fname in sorted(os.listdir(rollout_dir)):
        match = _STEP_FILE_RE.match(fname)
        if match:
            step = int(match.group(1))
            step_files.append((step, os.path.join(rollout_dir, fname)))
        elif fname.endswith(".jsonl"):
            warnings.append({
                "type": "invalid_filename",
                "filename": fname,
                "reason": "Filename does not match <step>.jsonl pattern",
            })
    
    step_files.sort(key=lambda x: x[0])
    return step_files, warnings


def _parse_single_record(
    record: dict[str, Any],
    experiment_id: str,
    step: int,
    epoch: Optional[int],
    generation_attempt: Optional[int],
    rollout_idx: int,
    group_id: str,
    source_file: str,
    source_line: int,
    record_idx: int = 0,
) -> RolloutRecord:
    """Parse a single JSONL record into a RolloutRecord."""
    # Extract fields
    prompt = _extract_str_field(record, ["input", "prompt"], record_idx)
    response = _extract_str_field(record, ["output", "response"], record_idx)
    reward = _extract_reward(record, record_idx)
    advantage = _extract_advantage(record, record_idx)
    advantage_estimator = _extract_str_field(record, ["adv_estimator", "advantage_estimator"], record_idx)
    reward_components = _extract_object_field(record, "reward_components", record_idx)
    selected_for_update = _extract_bool_field(record, ["selected_for_update"], record_idx)
    response_mask_tokens = _extract_int_field(record, "response_mask_tokens", record_idx)
    policy_signal_tokens = _extract_int_field(record, "policy_signal_tokens", record_idx)
    prediction = _extract_str_field(record, ["pred", "prediction"], record_idx)
    verifier_correct = _extract_bool_field(record, ["acc", "correct", "is_correct"], record_idx)
    ground_truth = _extract_str_field(record, ["gts", "ground_truth"], record_idx)
    difficulty = _extract_str_field(record, ["difficulty"], record_idx)
    source_dataset = _extract_str_field(record, ["data_source", "source_dataset"], record_idx)
    finish_reason = _extract_str_field(record, ["finish_reason"], record_idx)
    
    # Prompt ID
    prompt_id = _extract_str_field(record, ["prompt_id", "data_id", "id"])
    id_type = "original" if prompt_id else "unknown"
    
    # If no prompt_id but has prompt, generate content fingerprint
    if not prompt_id and prompt:
        prompt_id = content_fingerprint(prompt)
        id_type = "content_fingerprint"
    
    # Response token length (if response available)
    response_token_length = None
    if "response_token_length" in record:
        rtl = record["response_token_length"]
        if isinstance(rtl, (int, float)):
            response_token_length = int(rtl)
    elif response:
        response_token_length = len(response.split())  # rough estimate
    
    max_response_length = record.get("max_response_length")
    if max_response_length is not None:
        max_response_length = int(max_response_length)
    
    # Step cross-check
    _, step_warning = _extract_step(record, step)
    parse_status = "ok"
    parse_error = step_warning
    
    # Check for schema errors
    if response is not None and reward is None:
        if not _is_finite_number(record.get("score", record.get("reward"))):
            # Reward is missing or invalid
            pass  # Will be handled by quality analyzer
    
    # Check response/reward array length mismatch
    for resp_key in ("response", "output"):
        for rew_key in ("reward", "score"):
            if resp_key in record and rew_key in record:
                if isinstance(record[resp_key], list) and isinstance(record[rew_key], list):
                    if len(record[resp_key]) != len(record[rew_key]):
                        parse_status = "schema_error"
                        parse_error = f"response/reward length mismatch: {len(record[resp_key])} vs {len(record[rew_key])}"
    
    return RolloutRecord(
        schema_version=SCHEMA_VERSION,
        experiment_id=experiment_id,
        step=step,
        epoch=epoch,
        generation_attempt=generation_attempt,
        group_id=group_id,
        rollout_id=rollout_idx,
        prompt_id=prompt_id,
        id_type=id_type,
        prompt=prompt,
        response=response,
        training_reward=reward,
        reward_source="training",
        advantage=advantage,
        advantage_estimator=advantage_estimator,
        reward_components=reward_components,
        selected_for_update=selected_for_update,
        response_mask_tokens=response_mask_tokens,
        policy_signal_tokens=policy_signal_tokens,
        prediction=prediction,
        verifier_correct=verifier_correct,
        ground_truth=ground_truth,
        difficulty=difficulty,
        source_dataset=source_dataset,
        record_phase="train",
        source_file=source_file,
        source_line=source_line,
        parse_status=parse_status,
        parse_error=parse_error,
        finish_reason=finish_reason,
        response_token_length=response_token_length,
        max_response_length=max_response_length,
    )


def parse_rollout_dir(
    rollout_dir: str,
    experiment_id: str,
    epoch_map: Optional[dict[int, int]] = None,
) -> tuple[list[RolloutRecord], list[GroupRecord], list[dict[str, Any]]]:
    """Parse all rollout JSONL files in a directory.
    
    Args:
        rollout_dir: Directory containing `<step>.jsonl` files.
        experiment_id: Experiment identifier.
        epoch_map: Optional mapping from step to epoch.
    
    Returns:
        Tuple of (rollout_records, group_records, warnings).
    """
    if epoch_map is None:
        epoch_map = {}
    
    step_files, warnings = scan_rollout_dir(rollout_dir)
    
    all_rollouts: list[RolloutRecord] = []
    all_groups: list[GroupRecord] = []
    
    for step, filepath in step_files:
        epoch = epoch_map.get(step)
        
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        # Group records by group_id within this file
        step_rollouts: list[RolloutRecord] = []
        group_data: dict[str, list[RolloutRecord]] = {}
        
        for line_idx, raw_line in enumerate(lines, 1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            
            try:
                record = json.loads(raw_line)
                if not isinstance(record, dict):
                    raise ValueError(f"Expected JSON object, got {type(record).__name__}")
            except (json.JSONDecodeError, ValueError) as e:
                # Record parse error
                error_rollout = RolloutRecord(
                    schema_version=SCHEMA_VERSION,
                    experiment_id=experiment_id,
                    step=step,
                    epoch=epoch,
                    generation_attempt=None,
                    group_id="",
                    rollout_id=line_idx - 1,
                    prompt_id=None,
                    prompt=None,
                    response=None,
                    training_reward=None,
                    ground_truth=None,
                    source_file=filepath,
                    source_line=line_idx,
                    parse_status="json_error",
                    parse_error=str(e),
                )
                all_rollouts.append(error_rollout)
                warnings.append({
                    "type": "json_decode_error",
                    "file": filepath,
                    "line": line_idx,
                    "error": str(e),
                })
                continue
            
            # Determine group_id
            group_id = record.get("uid", record.get("group_id", ""))
            if not group_id:
                # Try to infer from prompt text (only within same file)
                prompt_text = _extract_str_field(record, ["input", "prompt"])
                if prompt_text:
                    group_id = content_fingerprint(prompt_text)
                else:
                    group_id = f"unknown_step{step}_line{line_idx}"
            
            # Determine rollout_id within group
            if group_id not in group_data:
                group_data[group_id] = []
            rollout_idx = len(group_data[group_id])
            
            # Determine generation_attempt
            gen_attempt = record.get("generation_attempt")
            if gen_attempt is not None:
                gen_attempt = int(gen_attempt)
            
            rollout = _parse_single_record(
                record=record,
                experiment_id=experiment_id,
                step=step,
                epoch=epoch,
                generation_attempt=gen_attempt,
                rollout_idx=rollout_idx,
                group_id=str(group_id),
                source_file=filepath,
                source_line=line_idx,
            )
            
            # Handle array-form records (single record with multiple responses)
            for resp_key in ("response", "output"):
                if resp_key in record and isinstance(record[resp_key], list):
                    responses = record[resp_key]
                    rewards_list = record.get("reward", record.get("score", []))
                    if not isinstance(rewards_list, list):
                        rewards_list = [rewards_list]
                    
                    if len(responses) != len(rewards_list):
                        rollout.parse_status = "schema_error"
                        rollout.parse_error = f"Array length mismatch: {len(responses)} responses vs {len(rewards_list)} rewards"
                    
                    # Expand array form into multiple rollouts
                    for arr_idx in range(len(responses)):
                        arr_rollout = _parse_single_record(
                            record=record,
                            experiment_id=experiment_id,
                            step=step,
                            epoch=epoch,
                            generation_attempt=gen_attempt,
                            rollout_idx=arr_idx,
                            group_id=str(group_id),
                            source_file=filepath,
                            source_line=line_idx,
                            record_idx=arr_idx,
                        )
                        arr_rollout.response = str(responses[arr_idx]) if arr_idx < len(responses) else None
                        arr_rollout.prediction = _extract_str_field(record, ["pred", "prediction"], arr_idx)
                        arr_rollout.verifier_correct = _extract_bool_field(
                            record, ["acc", "correct", "is_correct"], arr_idx
                        )
                        if arr_idx < len(rewards_list) and _is_finite_number(rewards_list[arr_idx]):
                            arr_rollout.training_reward = float(rewards_list[arr_idx])
                        else:
                            arr_rollout.training_reward = None
                        step_rollouts.append(arr_rollout)
                        group_data[group_id].append(arr_rollout)
                    break  # Only process one of response/output
            else:
                # Scalar form
                step_rollouts.append(rollout)
                group_data[group_id].append(rollout)
        
        all_rollouts.extend(step_rollouts)
        
        # Build group records
        for gid, rollouts in group_data.items():
            rewards = [r.training_reward for r in rollouts if r.training_reward is not None]
            advantages = [r.advantage for r in rollouts]
            selected_for_update = [r.selected_for_update for r in rollouts]
            responses = [r.response for r in rollouts]
            rollout_ids = [r.rollout_id for r in rollouts]
            
            # Get shared fields from first rollout
            first = rollouts[0] if rollouts else None
            group = GroupRecord(
                schema_version=SCHEMA_VERSION,
                experiment_id=experiment_id,
                step=step,
                epoch=epoch,
                generation_attempt=first.generation_attempt if first else None,
                group_id=gid,
                prompt_id=first.prompt_id if first else None,
                prompt=first.prompt if first else None,
                source_dataset=first.source_dataset if first else None,
                difficulty=first.difficulty if first else None,
                ground_truth=first.ground_truth if first else None,
                rollout_count=len(rollouts),
                rewards=rewards,
                advantages=advantages,
                advantage_estimator=first.advantage_estimator if first else None,
                selected_for_update=selected_for_update,
                responses=responses,
                rollout_ids=rollout_ids,
                source_file=filepath,
                group_status="resolved",
            )
            all_groups.append(group)
    
    return all_rollouts, all_groups, warnings
