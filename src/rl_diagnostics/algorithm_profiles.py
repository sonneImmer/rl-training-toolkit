"""Algorithm profiles used to explain observed training evidence."""
from __future__ import annotations


def resolve_algorithm_profile(algorithm: str | None, advantage_estimator: str | None = None) -> dict:
    algorithm_name = str(algorithm or "").strip().lower()
    estimator = str(advantage_estimator or "").strip().lower()
    effective = estimator or algorithm_name

    if algorithm_name == "ppo" or effective == "gae":
        return {
            "profile_id": "ppo_gae_v1",
            "family": "ppo",
            "display_name": "PPO (GAE)",
            "advantage_estimator": "gae",
            "group_relative_advantage": False,
            "mechanism": "Advantage comes from token rewards and critic values through GAE.",
        }
    if effective == "ngrpo" or algorithm_name == "ngrpo":
        return {
            "profile_id": "ngrpo_virtual_max_v1",
            "family": "ngrpo",
            "display_name": "NGRPO",
            "advantage_estimator": "ngrpo",
            "group_relative_advantage": True,
            "mechanism": "The current patch adds virtual maximum-reward references during group advantage normalization.",
        }
    if effective == "grpo" or algorithm_name == "grpo":
        return {
            "profile_id": "grpo_outcome_v1",
            "family": "grpo",
            "display_name": "GRPO",
            "advantage_estimator": "grpo",
            "group_relative_advantage": True,
            "mechanism": "Outcome rewards are normalized within each prompt group.",
        }
    return {
        "profile_id": "unknown",
        "family": "unknown",
        "display_name": algorithm or advantage_estimator or "Unknown",
        "advantage_estimator": advantage_estimator,
        "group_relative_advantage": None,
        "mechanism": "The active advantage mechanism was not identified from recorded configuration.",
    }
