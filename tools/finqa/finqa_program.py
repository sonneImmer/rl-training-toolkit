"""Dependency-free parsing and execution for FinQA programs."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


ALLOWED_OPS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "exp",
    "greater",
    "table_max",
    "table_min",
    "table_sum",
    "table_average",
}
PROGRAM_RE = re.compile(r"(?im)^\s*Program\s*:\s*(.+?)\s*$")
ANSWER_RE = re.compile(r"(?im)^\s*Answer\s*:\s*(.+?)\s*$")
EVIDENCE_RE = re.compile(r"(?im)^\s*Evidence\s*:\s*(.*?)\s*$")
END_RE = re.compile(r"(?im)^\s*End\s*:\s*(.*?)\s*$")
CALL_RE = re.compile(r"([a-z_]+)\s*\(([^()]*)\)", re.IGNORECASE)


@dataclass(frozen=True)
class ProgramResult:
    valid: bool
    value: Any = None
    error: str = ""
    steps: int = 0


def parse_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not numeric")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite number")
        return number

    text = str(value).strip().lower().replace(",", "").replace("$", "")
    if "(" in text and not text.startswith("("):
        text = text.split("(", 1)[0].strip()
    if text.startswith("const_"):
        text = text.removeprefix("const_").replace("m1", "-1")
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    percent = text.endswith("%")
    if percent:
        text = text[:-1].strip()
    number = float(text)
    if negative:
        number = -number
    if percent:
        number /= 100.0
    if not math.isfinite(number):
        raise ValueError("non-finite number")
    return number


def parse_output(text: str) -> dict[str, Any]:
    value = str(text or "")
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    answer_match = ANSWER_RE.search(value)
    program_match = PROGRAM_RE.search(value)
    evidence_match = EVIDENCE_RE.search(value)
    end_match = END_RE.search(value)
    evidence = []
    if evidence_match:
        evidence = [item.strip() for item in evidence_match.group(1).split(",") if item.strip()]
    format_valid = (
        len(lines) == 4
        and lines[0].lower().startswith("answer:")
        and lines[1].lower().startswith("program:")
        and lines[2].lower().startswith("evidence:")
        and lines[3] == "End: FINQA_END"
    )
    return {
        "answer": answer_match.group(1).strip() if answer_match else None,
        "program": program_match.group(1).strip() if program_match else None,
        "evidence": evidence,
        "end": end_match.group(1).strip() if end_match else None,
        "format_valid": format_valid,
    }


def parse_program(program: str) -> list[tuple[str, str, str]]:
    text = str(program or "").strip().rstrip(";")
    if not text:
        raise ValueError("missing program")
    calls = []
    cursor = 0
    for match in CALL_RE.finditer(text):
        separator = text[cursor:match.start()].strip()
        if separator not in ("", ",", ";"):
            raise ValueError(f"unexpected program text: {separator}")
        operation = match.group(1).lower()
        if operation not in ALLOWED_OPS:
            raise ValueError(f"unsupported operation: {operation}")
        arguments = [item.strip() for item in match.group(2).split(",", 1)]
        if len(arguments) != 2 or not all(arguments):
            raise ValueError(f"operation requires two arguments: {match.group(0)}")
        calls.append((operation, arguments[0], arguments[1]))
        cursor = match.end()
    if text[cursor:].strip() not in ("", ",", ";"):
        raise ValueError(f"unexpected program suffix: {text[cursor:].strip()}")
    if not calls:
        raise ValueError("no executable operations")
    return calls


def _table_rows(table: list[list[Any]]) -> dict[str, list[float]]:
    rows = {}
    for row in table[1:]:
        if not row:
            continue
        values = []
        for cell in row[1:]:
            try:
                values.append(parse_number(cell))
            except (TypeError, ValueError):
                continue
        if values:
            rows[str(row[0]).strip().lower()] = values
    return rows


def _resolve(argument: str, results: list[Any]) -> Any:
    if argument.startswith("#"):
        index = int(argument[1:])
        if index < 0 or index >= len(results):
            raise ValueError(f"invalid result reference: {argument}")
        return results[index]
    return parse_number(argument)


def execute_program(program: str, table: list[list[Any]]) -> ProgramResult:
    try:
        calls = parse_program(program)
        table_rows = _table_rows(table)
        results = []
        for operation, arg1_text, arg2_text in calls:
            if operation.startswith("table_"):
                row = table_rows.get(arg1_text.strip().lower())
                if not row:
                    raise ValueError(f"table row not found: {arg1_text}")
                if operation == "table_max":
                    value = max(row)
                elif operation == "table_min":
                    value = min(row)
                elif operation == "table_sum":
                    value = sum(row)
                else:
                    value = sum(row) / len(row)
            else:
                arg1 = _resolve(arg1_text, results)
                arg2 = _resolve(arg2_text, results)
                if operation == "add":
                    value = arg1 + arg2
                elif operation == "subtract":
                    value = arg1 - arg2
                elif operation == "multiply":
                    value = arg1 * arg2
                elif operation == "divide":
                    if arg2 == 0:
                        raise ValueError("division by zero")
                    value = arg1 / arg2
                elif operation == "exp":
                    value = arg1**arg2
                else:
                    value = "yes" if arg1 > arg2 else "no"
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("non-finite result")
            results.append(value)
        value = results[-1]
        if isinstance(value, float):
            value = round(value, 5)
        return ProgramResult(True, value=value, steps=len(results))
    except (ArithmeticError, OverflowError, TypeError, ValueError) as error:
        return ProgramResult(False, error=str(error))


def values_equal(left: Any, right: Any, tolerance: float = 1e-4) -> bool:
    if left is None or right is None:
        return False
    left_text = str(left).strip().lower()
    right_text = str(right).strip().lower()
    if left_text in {"yes", "no"} or right_text in {"yes", "no"}:
        return left_text == right_text
    try:
        left_number = parse_number(left)
        right_number = parse_number(right)
    except (TypeError, ValueError):
        return left_text == right_text
    scale = max(1.0, abs(left_number), abs(right_number))
    return abs(left_number - right_number) <= tolerance * scale
