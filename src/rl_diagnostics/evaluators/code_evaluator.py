"""REQ-10: Code evaluation adapter.

Executes generated code in an isolated subprocess with resource limits.
Returns 'blocked' if a safe sandbox is not available.

Supports HumanEval and MBPP style code evaluation with test cases.
Pure Python standard library only.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Configuration ────────────────────────────────────────────────

@dataclass
class CodeEvalConfig:
    """Configuration for code evaluation."""
    dataset_type: str = "auto"  # auto | humaneval | mbpp | generic
    pass_k: Optional[int] = None
    timeout_seconds: float = 10.0
    memory_limit_mb: int = 512
    cpu_time_limit_seconds: int = 10
    enable_sandbox: bool = True
    reward_threshold: Optional[float] = None
    config_version: str = "0.1.0-exp"

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class CodeSampleResult:
    """Per-sample evaluation result."""
    index: int = 0
    dataset_type: str = ""
    extracted_code: Optional[str] = None
    status: str = "correct"  # correct | extraction_failure | compile_error | runtime_error | test_failed | timeout | unsafe_api | blocked
    test_results: list[dict] = field(default_factory=list)
    test_pass_count: int = 0
    test_total_count: int = 0
    test_pass_rate: float = 0.0
    reward: Optional[float] = None
    reward_agree: Optional[bool] = None
    error_detail: str = ""
    unsafe_apis_detected: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class CodeEvalResult:
    """Full code evaluation result."""
    config_version: str = "0.1.0-exp"
    dataset_type: str = ""
    total_samples: int = 0
    extraction_success_count: int = 0
    extraction_success_rate: float = 0.0
    compile_success_count: int = 0
    compile_success_rate: float = 0.0
    runtime_success_count: int = 0
    runtime_success_rate: float = 0.0
    test_pass_rate: float = 0.0
    pass_at_1: float = 0.0
    pass_at_k: Optional[float] = None
    pass_k_value: Optional[int] = None
    timeout_count: int = 0
    timeout_rate: float = 0.0
    exception_count: int = 0
    exception_rate: float = 0.0
    unsafe_api_count: int = 0
    unsafe_api_rate: float = 0.0
    blocked: bool = False
    blocked_reason: str = ""
    reward_eval_agreement: Optional[float] = None
    reward_eval_disagreement_count: int = 0
    badcase_counts: dict[str, int] = field(default_factory=dict)
    samples: list[CodeSampleResult] = field(default_factory=list)
    badcases: list[CodeSampleResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# ── Code extraction ──────────────────────────────────────────────

_CODE_BLOCK_PATTERN = re.compile(r"```(?:python|Python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(response: str, dataset_type: str = "auto") -> Optional[str]:
    """Extract code from model response.

    Supports markdown code blocks and raw code.
    """
    # Try markdown code block
    matches = _CODE_BLOCK_PATTERN.findall(response)
    if matches:
        return matches[-1].strip()

    # HumanEval: response is the completion after a function signature
    # The function signature is in the prompt, response is the body
    # For standalone responses, check if it looks like code
    stripped = response.strip()
    if stripped:
        # Check if it contains Python keywords suggesting it's code
        code_indicators = ["def ", "import ", "class ", "return ", "if ", "for ", "while "]
        if any(ind in stripped for ind in code_indicators):
            return stripped

    return None


# ── Unsafe API detection ────────────────────────────────────────

_UNSAFE_PATTERNS = [
    (r"\bos\.system\s*\(", "os.system"),
    (r"\bsubprocess\.", "subprocess"),
    (r"\bos\.popen\s*\(", "os.popen"),
    (r"\b__import__\s*\(", "__import__"),
    (r"\beval\s*\(", "eval"),
    (r"\bexec\s*\(", "exec"),
    (r"\bopen\s*\(.*['\"]w", "file write"),
    (r"\bsocket\.", "socket"),
    (r"\bshutil\.rmtree", "shutil.rmtree"),
    (r"\bos\.remove", "os.remove"),
    (r"\bctypes\.", "ctypes"),
    (r"\bpickle\.loads", "pickle.loads"),
]


def detect_unsafe_apis(code: str) -> list[str]:
    """Detect potentially unsafe API usage in code."""
    detected = []
    for pattern, name in _UNSAFE_PATTERNS:
        if re.search(pattern, code):
            detected.append(name)
    return detected


# ── Syntax check ─────────────────────────────────────────────────

def check_syntax(code: str) -> tuple[bool, str]:
    """Check if code compiles without syntax errors.

    Returns (success, error_message).
    """
    try:
        ast.parse(code)
        compile(code, "<generated>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ── Sandbox execution ────────────────────────────────────────────

_SANDBOX_AVAILABLE = None


def check_sandbox_available() -> tuple[bool, str]:
    """Check if a safe execution sandbox is available.

    Returns (available, reason).
    """
    global _SANDBOX_AVAILABLE
    if _SANDBOX_AVAILABLE is not None:
        return _SANDBOX_AVAILABLE

    # Check if we can run subprocesses (basic isolation)
    try:
        result = subprocess.run(
            [sys.executable, "-c", "print('ok')"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip() == "ok":
            _SANDBOX_AVAILABLE = (True, "subprocess sandbox available")
            return _SANDBOX_AVAILABLE
    except Exception as e:
        _SANDBOX_AVAILABLE = (False, f"subprocess execution failed: {e}")
        return _SANDBOX_AVAILABLE

    _SANDBOX_AVAILABLE = (False, "subprocess execution not available")
    return _SANDBOX_AVAILABLE


def _build_test_script(code: str, test_cases: list[dict], dataset_type: str) -> str:
    """Build a test script that runs the generated code against test cases."""
    # Test cases format: [{"input": ..., "expected": ...}] or
    # [{"func_call": "func(args)", "expected": result}]
    script_lines = [
        "import sys",
        "import json",
        "import resource",
        "",
        "# Set resource limits",
        f"resource.setrlimit(resource.RLIMIT_CPU, ({_get_cpu_limit()}, {int(_get_cpu_limit()) + 1}))",
        f"resource.setrlimit(resource.RLIMIT_AS, ({_get_mem_limit()}, {int(_get_mem_limit()) + 1}))",
        "",
        "results = []",
        "",
        "# User code",
        "try:",
    ]
    # Indent user code
    for line in code.split("\n"):
        script_lines.append(f"    {line}" if line else "    ")
    script_lines.append("except Exception as e:")
    script_lines.append("    print(json.dumps({'runtime_error': str(e)}))")
    script_lines.append("    sys.exit(1)")
    script_lines.append("")
    script_lines.append("# Run tests")
    script_lines.append("for i, tc in enumerate(json.loads(sys.argv[1] if len(sys.argv) > 1 else '[]')):")
    script_lines.append("    try:")
    if dataset_type == "humaneval":
        script_lines.append("        result = eval(tc['func_call'])")
    else:
        script_lines.append("        result = eval(tc.get('func_call', tc.get('call', '')))")
    script_lines.append("        expected = tc['expected']")
    script_lines.append("        passed = (result == expected)")
    script_lines.append("        results.append({'test': i, 'passed': passed, 'got': repr(result), 'expected': repr(expected)})")
    script_lines.append("    except Exception as e:")
    script_lines.append("        results.append({'test': i, 'passed': False, 'error': str(e)})")
    script_lines.append("")
    script_lines.append("print(json.dumps(results))")

    return "\n".join(script_lines)


_CPU_LIMIT = 10
_MEM_LIMIT = 512 * 1024 * 1024


def _get_cpu_limit() -> int:
    return _CPU_LIMIT


def _get_mem_limit() -> int:
    return _MEM_LIMIT


def run_in_sandbox(
    code: str,
    test_cases: list[dict],
    config: CodeEvalConfig,
    dataset_type: str = "generic",
) -> tuple[str, list[dict], str]:
    """Run code in sandboxed subprocess.

    Returns (status, test_results, error_detail).
    """
    if not config.enable_sandbox:
        return "blocked", [], "Sandbox disabled in config"

    available, reason = check_sandbox_available()
    if not available:
        return "blocked", [], f"Sandbox not available: {reason}"

    # Build test script
    script = _build_test_script(code, test_cases, dataset_type)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        import json
        test_json = json.dumps(test_cases)
        result = subprocess.run(
            [sys.executable, script_path, test_json],
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )

        if result.returncode == 0:
            try:
                test_results = json.loads(result.stdout.strip())
                passed = all(t.get("passed", False) for t in test_results) if test_results else False
                if passed:
                    return "correct", test_results, ""
                else:
                    return "test_failed", test_results, "Some test cases failed"
            except json.JSONDecodeError:
                return "runtime_error", [], f"Could not parse output: {result.stdout[:200]}"
        else:
            output = result.stderr.strip()
            if "MemoryError" in output or "RLIMIT" in output:
                return "timeout", [], "Memory limit exceeded"
            if "CPU" in output:
                return "timeout", [], "CPU time limit exceeded"
            return "runtime_error", [], output[:500]

    except subprocess.TimeoutExpired:
        return "timeout", [], f"Execution timed out after {config.timeout_seconds}s"
    except Exception as e:
        return "runtime_error", [], f"Sandbox error: {e}"
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


# ── Main evaluation ──────────────────────────────────────────────

def evaluate_code(
    responses: list[str],
    test_cases: list[list[dict]],
    rewards: Optional[list[float]] = None,
    config: Optional[CodeEvalConfig] = None,
) -> CodeEvalResult:
    """Evaluate generated code against test cases.

    Args:
        responses: Model-generated code responses (one per sample for pass@1,
                   or list of lists for pass@k).
        test_cases: List of test case lists. Each element is a list of
                    test case dicts for one problem.
        rewards: Optional RL training rewards for agreement check.
        config: Evaluation configuration.

    Returns:
        CodeEvalResult with all computed metrics.
    """
    if config is None:
        config = CodeEvalConfig()

    result = CodeEvalResult(
        config_version=config.config_version,
        dataset_type=config.dataset_type,
    )
    result.total_samples = len(responses)

    if result.total_samples == 0:
        result.blocked = True
        result.blocked_reason = "No samples provided"
        return result

    # Check sandbox availability upfront
    available, reason = check_sandbox_available()
    if not available:
        result.blocked = True
        result.blocked_reason = reason
        # Still do extraction and syntax check
        for i, (resp, tcs) in enumerate(zip(responses, test_cases)):
            sr = CodeSampleResult(index=i, status="blocked")
            code = extract_code(resp, config.dataset_type)
            sr.extracted_code = code
            if code:
                result.extraction_success_count += 1
                ok, err = check_syntax(code)
                if ok:
                    result.compile_success_count += 1
            result.samples.append(sr)
        result.extraction_success_rate = result.extraction_success_count / result.total_samples
        result.compile_success_rate = result.compile_success_count / result.total_samples
        return result

    is_pass_k = bool(responses) and isinstance(responses[0], list)

    for i, (resp, tcs) in enumerate(zip(responses, test_cases)):
        sr = CodeSampleResult(index=i, dataset_type=config.dataset_type)
        sr.reward = rewards[i] if rewards else None

        # Extract code
        single_resp = resp[0] if is_pass_k else resp
        code = extract_code(single_resp, config.dataset_type)
        sr.extracted_code = code

        if code is None:
            sr.status = "extraction_failure"
            sr.error_detail = "Could not extract code from response"
        else:
            result.extraction_success_count += 1

            # Check for unsafe APIs
            unsafe = detect_unsafe_apis(code)
            sr.unsafe_apis_detected = unsafe
            if unsafe:
                sr.status = "unsafe_api"
                sr.error_detail = f"Unsafe APIs detected: {', '.join(unsafe)}"
            else:
                # Syntax check
                ok, err = check_syntax(code)
                if not ok:
                    sr.status = "compile_error"
                    sr.error_detail = err
                else:
                    result.compile_success_count += 1

                    # Run in sandbox
                    status, test_results, error = run_in_sandbox(
                        code, tcs, config, config.dataset_type
                    )
                    sr.test_results = test_results
                    sr.status = status
                    sr.error_detail = error

                    if status == "correct":
                        sr.test_pass_count = len(test_results)
                        sr.test_total_count = len(test_results)
                        sr.test_pass_rate = 1.0
                    elif status == "test_failed":
                        sr.test_pass_count = sum(1 for t in test_results if t.get("passed"))
                        sr.test_total_count = len(test_results)
                        sr.test_pass_rate = sr.test_pass_count / sr.test_total_count if sr.test_total_count else 0.0
                        result.runtime_success_count += 1
                    elif status == "timeout":
                        result.timeout_count += 1
                    elif status == "runtime_error":
                        result.exception_count += 1
                        result.runtime_success_count += 1
                    elif status == "blocked":
                        result.blocked = True
                        result.blocked_reason = error

        result.samples.append(sr)
        if sr.status != "correct":
            result.badcases.append(sr)

    # Compute aggregate metrics
    total = result.total_samples
    result.extraction_success_rate = result.extraction_success_count / total if total else 0.0
    result.compile_success_rate = result.compile_success_count / total if total else 0.0
    result.runtime_success_rate = result.runtime_success_count / total if total else 0.0
    result.timeout_rate = result.timeout_count / total if total else 0.0
    result.exception_rate = result.exception_count / total if total else 0.0
    result.unsafe_api_count = sum(1 for s in result.samples if s.unsafe_apis_detected)
    result.unsafe_api_rate = result.unsafe_api_count / total if total else 0.0

    # Test pass rate (average across all samples)
    pass_rates = [s.test_pass_rate for s in result.samples]
    result.test_pass_rate = sum(pass_rates) / total if total else 0.0

    # Pass@1
    correct_count = sum(1 for s in result.samples if s.status == "correct")
    result.pass_at_1 = correct_count / total if total else 0.0

    # Badcase counts
    for s in result.samples:
        if s.status != "correct":
            result.badcase_counts[s.status] = result.badcase_counts.get(s.status, 0) + 1

    # Pass@k
    if is_pass_k and config.pass_k and config.pass_k > 1:
        result.pass_k_value = config.pass_k
        result.pass_at_k = _compute_code_pass_at_k(
            responses, test_cases, config, k=config.pass_k
        )

    # Reward-eval agreement
    if rewards is not None:
        r_thresh = config.reward_threshold if config.reward_threshold is not None else 0.5
        agree = 0
        for s in result.samples:
            if s.reward is not None:
                reward_correct = s.reward >= r_thresh
                eval_correct = s.status == "correct"
                s.reward_agree = (reward_correct == eval_correct)
                if s.reward_agree:
                    agree += 1
                else:
                    result.reward_eval_disagreement_count += 1
        result.reward_eval_agreement = agree / total if total else 0.0

    return result


def _compute_code_pass_at_k(
    responses: list[list[str]],
    test_cases: list[list[dict]],
    config: CodeEvalConfig,
    k: int,
) -> float:
    """Compute unbiased pass@k for code evaluation."""
    import functools

    @functools.lru_cache(maxsize=1)
    def comb(n: int, r: int) -> int:
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(r):
            result = result * (n - i) // (i + 1)
        return result

    total = len(responses)
    if total == 0:
        return 0.0

    pass_sum = 0.0
    for resp_list, tcs in zip(responses, test_cases):
        n = len(resp_list)
        if n == 0:
            continue
        kk = min(k, n)
        c = 0
        for resp in resp_list:
            code = extract_code(resp, config.dataset_type)
            if code is None:
                continue
            if detect_unsafe_apis(code):
                continue
            ok, _ = check_syntax(code)
            if not ok:
                continue
            status, _, _ = run_in_sandbox(code, tcs, config, config.dataset_type)
            if status == "correct":
                c += 1
        if c == 0:
            pass_sum += 0.0
        elif c >= n:
            pass_sum += 1.0
        else:
            pass_sum += 1.0 - comb(n - c, kk) / comb(n, kk)

    return pass_sum / total
