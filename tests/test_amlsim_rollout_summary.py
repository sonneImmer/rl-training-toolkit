import json
from pathlib import Path

from tools.amlsim.summarize_amlsim_rollouts import summarize


def test_summarize_tracks_typology_learning_signal(tmp_path: Path) -> None:
    rows = [
        {"step": 1, "input": "fan-in", "expected": "FAN_IN", "acc": True, "format_valid": True, "score": 4.0},
        {"step": 1, "input": "fan-in", "expected": "FAN_IN", "acc": False, "format_valid": True, "score": 2.0},
        {"step": 1, "input": "normal", "expected": "NORMAL", "acc": True, "format_valid": True, "score": 4.0},
        {"step": 1, "input": "normal", "expected": "NORMAL", "acc": True, "format_valid": True, "score": 4.0},
    ]
    path = tmp_path / "1.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = summarize(path)

    assert result["gradient_group_rate"] == 0.5
    assert result["per_typology"]["FAN_IN"]["exact_rate"] == 0.5
    assert result["per_typology"]["FAN_IN"]["gradient_group_rate"] == 1.0
    assert result["per_typology"]["NORMAL"]["all_correct"] == 1
