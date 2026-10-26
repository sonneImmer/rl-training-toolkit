import json

from tools.finqa.finqa_program import execute_program, parse_number, values_equal
from tools.finqa.finqa_reward import compute_score


TABLE = [
    ["", "2023", "2022"],
    ["revenue", "$ 120", "$ 100"],
    ["margin", "25% ( 25 % )", "20% ( 20 % )"],
]
TRUTH = json.dumps({
    "answer": 20.0,
    "gold_program": "subtract(120, 100)",
    "gold_evidence": ["table_1"],
})
EXTRA = {"table": json.dumps(TABLE)}


def reward(response):
    return compute_score("finqa_executable_reasoning", response, TRUTH, EXTRA)


def output(answer, program, evidence="table_1"):
    return f"Answer: {answer}\nProgram: {program}\nEvidence: {evidence}\nEnd: FINQA_END"


def test_program_executor_supports_financial_numbers_and_references():
    result = execute_program("subtract(120, 100), divide(#0, 100)", TABLE)
    assert result.valid is True
    assert values_equal(result.value, 0.2)
    assert parse_number("$ -1 ( 1 )") == -1.0
    assert parse_number("25% ( 25 % )") == 0.25


def test_program_executor_supports_table_operations():
    result = execute_program("table_average(margin, none)", TABLE)
    assert result.valid is True
    assert values_equal(result.value, 0.225)


def test_reward_full_score_requires_correct_execution_answer_and_evidence():
    result = reward(output("20", "subtract(120, 100)"))
    assert result["score"] == 4.0
    assert result["acc"] is True
    assert result["evidence_recall"] == 1.0


def test_reward_does_not_accept_a_guessed_answer_with_wrong_program():
    result = reward(output("20", "add(120, 100)"))
    assert result["score"] == -2.0
    assert result["execution_correct"] is False


def test_reward_marks_invalid_program():
    result = reward(output("20", "import(os, system)"))
    assert result["score"] == -3.0
    assert result["program_valid"] is False


def test_reward_detects_program_answer_inconsistency():
    result = reward(output("21", "subtract(120, 100)"))
    assert result["score"] == 1.0
    assert result["execution_correct"] is True
    assert result["answer_consistent"] is False


def test_reward_uses_evidence_only_as_a_small_bonus():
    result = reward(output("20", "subtract(120, 100)", "text_9"))
    assert result["score"] == 3.5
    assert result["acc"] is True
    assert result["evidence_recall"] == 0.0


def test_reward_rejects_truncated_or_runaway_output():
    truncated = reward("Answer: 20\nProgram: subtract(120, 100)\nEvidence: table_1")
    runaway = reward(output("20", "subtract(120, 100)") + "\nAnswer: 99")
    assert truncated["score"] == -3.5
    assert runaway["score"] == -3.5
    assert truncated["reward_reason"] == "invalid_output_format"


def test_reward_penalizes_extra_evidence_with_precision():
    result = reward(output("20", "subtract(120, 100)", "table_1, text_9"))
    assert result["score"] < 4.0
    assert result["evidence_recall"] == 1.0
    assert result["evidence_precision"] == 0.5
