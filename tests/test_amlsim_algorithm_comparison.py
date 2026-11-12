import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "amlsim" / "analyze_algorithm_comparison.py"
SPEC = importlib.util.spec_from_file_location("analyze_algorithm_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_distinction_checks_detect_expected_algorithm_evidence():
    rows = [
        {"algorithm": "grpo", "homogeneous_zero_advantage_groups": 5, "homogeneous_nonzero_advantage_groups": 0},
        {"algorithm": "ngrpo", "homogeneous_zero_advantage_groups": 0, "homogeneous_nonzero_advantage_groups": 4},
        {
            "algorithm": "ppo", "advantage_estimator": "gae",
            "metric_means": {"critic/vf_loss": 1.0, "timing_s/update_critic": 2.0},
        },
    ]

    checks = {item["name"]: item for item in MODULE.distinction_checks(rows)}

    assert checks["homogeneous_reward_signal_difference"]["detected"] is True
    assert checks["ppo_critic_path_detected"]["detected"] is True
