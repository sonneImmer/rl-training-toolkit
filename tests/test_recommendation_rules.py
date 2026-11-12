from types import SimpleNamespace

from rl_diagnostics.recommendation_rules import build_mechanism_diagnostics


def _group(**overrides):
    values = {
        "step": 1,
        "signal_type": "constant_score",
        "advantage_recorded_count": 4,
        "homogeneous_reward_zero_advantage": False,
        "homogeneous_reward_nonzero_advantage": False,
        "all_wrong_negative_advantage": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_grpo_diagnoses_homogeneous_zero_advantage():
    result = build_mechanism_diagnostics(
        "grpo", "grpo", [_group(homogeneous_reward_zero_advantage=True)], [],
    )
    assert result["algorithm_profile"]["family"] == "grpo"
    assert "group_advantage_collapse" in {card["rule_id"] for card in result["cards"]}


def test_ngrpo_diagnoses_nonzero_and_negative_advantage():
    result = build_mechanism_diagnostics(
        "ngrpo", "ngrpo", [
            _group(homogeneous_reward_nonzero_advantage=True, all_wrong_negative_advantage=True),
        ], [],
    )
    ids = {card["rule_id"] for card in result["cards"]}
    assert "ngrpo_homogeneous_reward_nonzero_advantage" in ids
    assert "ngrpo_negative_advantage_concentration" in ids


def test_gae_is_resolved_as_real_ppo_profile():
    result = build_mechanism_diagnostics("ppo", "gae", [_group(signal_type="effective_group")], [])
    assert result["algorithm_profile"]["family"] == "ppo"
    assert "ppo_gae_signal" in {card["rule_id"] for card in result["cards"]}
