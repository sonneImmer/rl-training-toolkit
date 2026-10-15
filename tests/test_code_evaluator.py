"""Tests for REQ-10: Code evaluation adapter."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_diagnostics.evaluators.code_evaluator import (
    extract_code, detect_unsafe_apis, check_syntax, evaluate_code,
    CodeEvalConfig, check_sandbox_available,
)


class TestCodeExtraction:
    def test_markdown_block(self):
        resp = "Here's the code:\n```python\ndef add(a, b):\n    return a + b\n```"
        code = extract_code(resp)
        assert code is not None
        assert "def add" in code

    def test_bare_code_block(self):
        resp = "```\ndef foo():\n    return 1\n```"
        code = extract_code(resp)
        assert code is not None
        assert "def foo" in code

    def test_raw_code(self):
        resp = "def solution():\n    return 42"
        code = extract_code(resp)
        assert code is not None
        assert "def solution" in code

    def test_no_code(self):
        resp = "I cannot solve this problem."
        code = extract_code(resp)
        assert code is None

    def test_multiple_blocks(self):
        resp = "```python\nimport os\n```\n```python\ndef main():\n    return 1\n```"
        code = extract_code(resp)
        assert "def main" in code


class TestUnsafeAPIDetection:
    def test_os_system(self):
        code = "import os\nos.system('rm -rf /')"
        assert "os.system" in detect_unsafe_apis(code)

    def test_subprocess(self):
        code = "import subprocess\nsubprocess.run(['ls'])"
        assert "subprocess" in detect_unsafe_apis(code)

    def test_eval(self):
        code = "eval('1+1')"
        assert "eval" in detect_unsafe_apis(code)

    def test_exec(self):
        code = "exec('print(1)')"
        assert "exec" in detect_unsafe_apis(code)

    def test_socket(self):
        code = "import socket\ns = socket.socket()"
        assert "socket" in detect_unsafe_apis(code)

    def test_safe_code(self):
        code = "def add(a, b):\n    return a + b"
        assert detect_unsafe_apis(code) == []

    def test_multiple_unsafe(self):
        code = "import os\nos.system('ls')\nexec('x=1')"
        detected = detect_unsafe_apis(code)
        assert "os.system" in detected
        assert "exec" in detected


class TestSyntaxCheck:
    def test_valid_code(self):
        ok, err = check_syntax("def foo():\n    return 1")
        assert ok is True
        assert err == ""

    def test_syntax_error(self):
        ok, err = check_syntax("def foo(\n    return 1")
        assert ok is False
        assert "SyntaxError" in err

    def test_empty_code(self):
        ok, err = check_syntax("")
        assert ok is True  # Empty is syntactically valid

    def test_indent_error(self):
        ok, err = check_syntax("def foo():\nreturn 1")
        assert ok is False


class TestSandboxCheck:
    def test_sandbox_check(self):
        available, reason = check_sandbox_available()
        # In test environment, subprocess should be available
        assert isinstance(available, bool)
        assert isinstance(reason, str)


class TestEvaluateCode:
    def test_correct_code(self):
        responses = ["def add(a, b):\n    return a + b"]
        test_cases = [[
            {"func_call": "add(1, 2)", "expected": 3},
            {"func_call": "add(0, 0)", "expected": 0},
        ]]
        result = evaluate_code(responses, test_cases)
        assert result.total_samples == 1
        assert result.extraction_success_rate == 1.0
        if not result.blocked:
            assert result.compile_success_rate == 1.0
            assert result.pass_at_1 == 1.0

    def test_extraction_failure(self):
        responses = ["I cannot solve this"]
        test_cases = [[{"func_call": "add(1,2)", "expected": 3}]]
        result = evaluate_code(responses, test_cases)
        assert result.extraction_success_rate == 0.0
        assert result.badcase_counts.get("extraction_failure") == 1

    def test_syntax_error(self):
        responses = ["def add(a, b)\n    return a + b"]
        test_cases = [[{"func_call": "add(1,2)", "expected": 3}]]
        result = evaluate_code(responses, test_cases)
        assert result.badcase_counts.get("compile_error", 0) >= 1

    def test_unsafe_api(self):
        responses = ["import os\ndef foo():\n    os.system('ls')\n    return 1"]
        test_cases = [[{"func_call": "foo()", "expected": 1}]]
        result = evaluate_code(responses, test_cases)
        assert result.unsafe_api_count == 1
        assert result.badcase_counts.get("unsafe_api") == 1

    def test_empty_input(self):
        result = evaluate_code([], [])
        assert result.blocked is True
        assert result.total_samples == 0

    def test_reward_agreement(self):
        responses = ["def add(a, b):\n    return a + b"]
        test_cases = [[{"func_call": "add(1,2)", "expected": 3}]]
        rewards = [1.0]
        result = evaluate_code(responses, test_cases, rewards=rewards)
        if not result.blocked:
            assert result.reward_eval_agreement is not None

    def test_blocked_sandbox_disabled(self):
        config = CodeEvalConfig(enable_sandbox=False)
        responses = ["def foo():\n    return 1"]
        test_cases = [[{"func_call": "foo()", "expected": 1}]]
        result = evaluate_code(responses, test_cases, config=config)
        assert result.blocked is True

    def test_test_failure(self):
        responses = ["def add(a, b):\n    return a - b"]
        test_cases = [[
            {"func_call": "add(1, 2)", "expected": 3},
        ]]
        result = evaluate_code(responses, test_cases)
        if not result.blocked:
            assert result.pass_at_1 == 0.0
            assert "test_failed" in result.badcase_counts

    def test_timeout(self):
        """Test that infinite loop triggers timeout."""
        responses = ["def loop():\n    while True:\n        pass\n    return 1"]
        test_cases = [[{"func_call": "loop()", "expected": 1}]]
        config = CodeEvalConfig(timeout_seconds=2)
        result = evaluate_code(responses, test_cases, config=config)
        if not result.blocked:
            assert result.timeout_count >= 1
            assert "timeout" in result.badcase_counts
