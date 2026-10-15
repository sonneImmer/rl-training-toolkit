"""REQ-09: Math evaluation adapter.

Supports GSM8K and MATH style answer extraction, numerical normalization,
and equivalence checking. Pure Python standard library only.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional


# ── Configuration ────────────────────────────────────────────────

@dataclass
class MathEvalConfig:
    """Configuration for math evaluation."""
    dataset_type: str = "auto"  # auto | gsm8k | math | generic
    tolerance: float = 1e-6
    pass_k: Optional[int] = None
    reward_threshold: Optional[float] = None
    config_version: str = "0.1.0-exp"

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class MathSampleResult:
    """Per-sample evaluation result."""
    index: int = 0
    dataset_type: str = ""
    extracted_answer: Optional[str] = None
    normalized_answer: Optional[str] = None
    ground_truth_raw: str = ""
    ground_truth_normalized: Optional[str] = None
    status: str = "correct"  # correct | parse_failure | format_error | answer_wrong | verifier_error
    exact_match: bool = False
    equivalent: bool = False
    reward: Optional[float] = None
    reward_agree: Optional[bool] = None
    error_detail: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class MathEvalResult:
    """Full math evaluation result."""
    config_version: str = "0.1.0-exp"
    dataset_type: str = ""
    total_samples: int = 0
    parse_success_count: int = 0
    parse_success_rate: float = 0.0
    exact_match_count: int = 0
    exact_match: float = 0.0
    equivalent_count: int = 0
    equivalent_accuracy: float = 0.0
    pass_at_1: float = 0.0
    pass_at_k: Optional[float] = None
    pass_k_value: Optional[int] = None
    reward_eval_agreement: Optional[float] = None
    reward_eval_disagreement_count: int = 0
    badcase_counts: dict[str, int] = field(default_factory=dict)
    degenerate: bool = False
    degenerate_reason: str = ""
    samples: list[MathSampleResult] = field(default_factory=list)
    badcases: list[MathSampleResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# ── Answer extraction ────────────────────────────────────────────

_GSM8K_PATTERN = re.compile(r"####\s*(.+)")
_BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_LAST_NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_gsm8k_answer(response: str) -> Optional[str]:
    """Extract answer after #### marker (GSM8K style)."""
    matches = _GSM8K_PATTERN.findall(response)
    if matches:
        return matches[-1].strip()
    # Fallback: last number
    nums = _LAST_NUMBER_PATTERN.findall(response)
    if nums:
        return nums[-1].strip()
    return None


def extract_math_answer(response: str) -> Optional[str]:
    """Extract answer from \\boxed{} (MATH style)."""
    matches = _BOXED_PATTERN.findall(response)
    if matches:
        return matches[-1].strip()
    # Fallback: "The answer is X" pattern
    m = re.search(r"(?:answer|Answer)\s*(?:is|=|:)\s*(.+?)(?:\.|$)", response)
    if m:
        return m.group(1).strip()
    # Last number fallback
    nums = _LAST_NUMBER_PATTERN.findall(response)
    if nums:
        return nums[-1].strip()
    return None


def extract_generic_answer(response: str) -> Optional[str]:
    """Generic extraction: last number in response."""
    nums = _LAST_NUMBER_PATTERN.findall(response)
    if nums:
        return nums[-1].strip()
    return None


def extract_answer(response: str, dataset_type: str = "auto") -> tuple[Optional[str], str]:
    """Extract answer from response. Returns (answer, detected_type)."""
    if dataset_type == "gsm8k":
        return extract_gsm8k_answer(response), "gsm8k"
    if dataset_type == "math":
        return extract_math_answer(response), "math"
    if dataset_type == "generic":
        return extract_generic_answer(response), "generic"
    # Auto-detect
    if "####" in response:
        return extract_gsm8k_answer(response), "gsm8k"
    if "\\boxed" in response:
        return extract_math_answer(response), "math"
    return extract_generic_answer(response), "generic"


# ── Normalization ────────────────────────────────────────────────

def _strip_latex(text: str) -> str:
    """Remove common LaTeX formatting."""
    text = text.strip()
    text = text.rstrip(".,;:!?")
    # Remove \text{...}
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    # Remove \mathrm{...}
    text = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text)
    # Remove \left and \right
    text = text.replace("\\left", "").replace("\\right", "")
    # Remove surrounding $...$
    if text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()
    # Remove \, spacing
    text = text.replace("\\,", "").replace("\\ ", "")
    # Remove \dfrac -> \frac
    text = text.replace("\\dfrac", "\\frac")
    # Remove \tfrac -> \frac
    text = text.replace("\\tfrac", "\\frac")
    return text.strip()


def _parse_latex_fraction(text: str) -> Optional[float]:
    """Parse \\frac{a}{b} or a/b into float."""
    m = re.match(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", text)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den != 0:
            return num / den
    # Simple a/b fraction
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$", text)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den != 0:
            return num / den
    return None


def _normalize_value(text: str) -> Optional[str]:
    """Normalize an answer string to a canonical form.

    Returns a string that can be compared for exact match,
    or None if normalization fails.
    """
    text = _strip_latex(text)
    if not text:
        return None

    # Remove commas in numbers (e.g., 1,000 -> 1000)
    text = re.sub(r"(\d),(\d)", r"\1\2", text)

    # Handle percentage
    if text.endswith("%"):
        val_str = text[:-1].strip()
        try:
            return str(float(val_str) / 100.0)
        except ValueError:
            pass

    # Handle LaTeX fraction
    frac_val = _parse_latex_fraction(text)
    if frac_val is not None:
        return str(frac_val)

    # Handle \pi
    if text in ("\\pi", "pi"):
        return str(math.pi)
    m = re.match(r"(-?\d+(?:\.\d+)?)\\?pi", text)
    if m:
        return str(float(m.group(1)) * math.pi)

    # Handle \sqrt{n}
    m = re.match(r"\\sqrt\{(\d+(?:\.\d+)?)\}", text)
    if m:
        return str(math.sqrt(float(m.group(1))))

    # Handle exponent: 2^3, 2^{3}
    m = re.match(r"(-?\d+(?:\.\d+)?)\^(\{?(\d+(?:\.\d+)?)\}?)", text)
    if m:
        return str(float(m.group(1)) ** float(m.group(3)))

    # Try plain float
    try:
        val = float(text)
        # Normalize: remove trailing .0
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        pass

    # Return cleaned text as-is for string comparison
    return text


def normalize_answer(text: str) -> Optional[str]:
    """Public normalization entry point."""
    if text is None:
        return None
    result = _normalize_value(text)
    return result


# ── Equivalence checking ─────────────────────────────────────────

def check_equivalent(
    pred_norm: Optional[str],
    gt_norm: Optional[str],
    tolerance: float = 1e-6,
) -> bool:
    """Check if two normalized answers are equivalent."""
    if pred_norm is None or gt_norm is None:
        return False
    # Exact string match
    if pred_norm == gt_norm:
        return True
    # Try numerical comparison
    try:
        p = float(pred_norm)
        g = float(gt_norm)
        if abs(p - g) <= tolerance:
            return True
        # Relative tolerance for large numbers
        if g != 0 and abs(p - g) / abs(g) <= tolerance:
            return True
        return False
    except ValueError:
        # Non-numeric: string comparison already done
        return False


# ── Main evaluation ──────────────────────────────────────────────

def evaluate_math(
    responses: list[str],
    ground_truths: list[str],
    rewards: Optional[list[float]] = None,
    config: Optional[MathEvalConfig] = None,
) -> MathEvalResult:
    """Evaluate math answers.

    Args:
        responses: Model-generated responses (one per sample for pass@1,
                   or list of lists for pass@k).
        ground_truths: Ground truth answer strings.
        rewards: Optional RL training rewards for agreement check.
        config: Evaluation configuration.

    Returns:
        MathEvalResult with all computed metrics.
    """
    if config is None:
        config = MathEvalConfig()

    result = MathEvalResult(
        config_version=config.config_version,
        dataset_type=config.dataset_type,
    )
    result.total_samples = len(responses)

    if result.total_samples == 0:
        result.degenerate = True
        result.degenerate_reason = "No samples provided"
        return result

    # Detect if this is pass@k (responses are lists)
    is_pass_k = bool(responses) and isinstance(responses[0], list)
    k = config.pass_k if config.pass_k else 1

    for i, (resp, gt) in enumerate(zip(responses, ground_truths)):
        sr = MathSampleResult(index=i, ground_truth_raw=gt)
        sr.reward = rewards[i] if rewards else None

        try:
            # For pass@k, take first response; we'll handle k separately
            single_resp = resp[0] if is_pass_k else resp
            extracted, dtype = extract_answer(single_resp, config.dataset_type)
            sr.dataset_type = dtype
            sr.extracted_answer = extracted

            if extracted is None:
                sr.status = "parse_failure"
                sr.error_detail = "Could not extract answer from response"
            else:
                gt_norm = normalize_answer(gt)
                pred_norm = normalize_answer(extracted)
                sr.ground_truth_normalized = gt_norm
                sr.normalized_answer = pred_norm

                if pred_norm is None:
                    sr.status = "format_error"
                    sr.error_detail = f"Could not normalize extracted answer: {extracted}"
                elif gt_norm is None:
                    sr.status = "format_error"
                    sr.error_detail = f"Could not normalize ground truth: {gt}"
                else:
                    sr.exact_match = (pred_norm == gt_norm)
                    sr.equivalent = check_equivalent(pred_norm, gt_norm, config.tolerance)
                    if sr.equivalent:
                        sr.status = "correct"
                    else:
                        sr.status = "answer_wrong"
                        sr.error_detail = f"pred={pred_norm} vs gt={gt_norm}"

        except Exception as e:
            sr.status = "verifier_error"
            sr.error_detail = str(e)

        result.samples.append(sr)
        if sr.status != "correct":
            result.badcases.append(sr)

    # Compute metrics
    total = result.total_samples
    result.parse_success_count = sum(1 for s in result.samples if s.status not in ("parse_failure",))
    result.parse_success_rate = result.parse_success_count / total if total else 0.0

    result.exact_match_count = sum(1 for s in result.samples if s.exact_match)
    result.exact_match = result.exact_match_count / total if total else 0.0

    result.equivalent_count = sum(1 for s in result.samples if s.equivalent)
    result.equivalent_accuracy = result.equivalent_count / total if total else 0.0

    # Badcase counts
    for s in result.samples:
        if s.status != "correct":
            result.badcase_counts[s.status] = result.badcase_counts.get(s.status, 0) + 1

    # Pass@1
    result.pass_at_1 = result.equivalent_accuracy

    # Pass@k (unbiased estimator)
    if is_pass_k and config.pass_k and config.pass_k > 1:
        result.pass_k_value = config.pass_k
        result.pass_at_k = _compute_pass_at_k(
            responses, ground_truths, config, k=config.pass_k
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


def _compute_pass_at_k(
    responses: list[list[str]],
    ground_truths: list[str],
    config: MathEvalConfig,
    k: int,
) -> float:
    """Compute unbiased pass@k estimator.

    For each problem with n samples where c are correct:
    pass@k = 1 - C(n-c, k) / C(n, k)
    """
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
    for resp_list, gt in zip(responses, ground_truths):
        n = len(resp_list)
        if n == 0:
            continue
        kk = min(k, n)
        # Count correct
        c = 0
        for r in resp_list:
            extracted, _ = extract_answer(r, config.dataset_type)
            if extracted is None:
                continue
            pred_norm = normalize_answer(extracted)
            gt_norm = normalize_answer(gt)
            if pred_norm and gt_norm and check_equivalent(pred_norm, gt_norm, config.tolerance):
                c += 1
        if c == 0:
            pass_sum += 0.0
        elif c >= n:
            pass_sum += 1.0
        else:
            # 1 - C(n-c, k) / C(n, k)
            pass_sum += 1.0 - comb(n - c, kk) / comb(n, kk)

    return pass_sum / total
