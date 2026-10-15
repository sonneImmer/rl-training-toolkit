"""REQ-08: Post-lending risk control evaluation adapter.

Computes classification metrics for imbalanced risk prediction tasks.
Uses only Python standard library — no sklearn dependency.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RiskEvalConfig:
    """Configuration for risk evaluation."""
    positive_label: Any = 1
    precision_floor: Optional[float] = None  # Recall@Precision >= P
    recall_floor: Optional[float] = None     # Precision@Recall >= R
    top_k: Optional[int] = None              # top-k recall / lift
    has_probability: bool = False             # model outputs probability/score
    reward_threshold: Optional[float] = None  # RL reward -> binary label threshold
    config_version: str = "0.1.0-exp"

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass
class RiskEvalResult:
    """Full risk evaluation result."""
    config_version: str = "0.1.0-exp"
    total_samples: int = 0
    positive_count: int = 0
    negative_count: int = 0
    positive_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    f2: float = 0.0
    pr_auc: float = 0.0
    roc_auc: float = 0.0
    ks: float = 0.0
    recall_at_precision: Optional[float] = None
    precision_at_recall: Optional[float] = None
    topk_recall: Optional[float] = None
    topk_lift: Optional[float] = None
    false_negative_count: int = 0
    false_negative_rate: float = 0.0
    brier_score: Optional[float] = None
    calibration_error: Optional[float] = None
    reward_eval_agreement: Optional[float] = None
    reward_eval_disagreement_count: int = 0
    degenerate: bool = False
    degenerate_reason: str = ""
    pr_curve: list[dict] = field(default_factory=list)
    false_negative_badcases: list[dict] = field(default_factory=list)
    reward_eval_mismatches: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# ── Core metric calculations ─────────────────────────────────────

def _confusion_counts(labels: list, preds: list) -> tuple[int, int, int, int]:
    """Return (TP, FP, TN, FN)."""
    tp = fp = tn = fn = 0
    for lbl, prd in zip(labels, preds):
        if lbl and prd:
            tp += 1
        elif not lbl and prd:
            fp += 1
        elif not lbl and not prd:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def _precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def _recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _fbeta(p: float, r: float, beta: float) -> float:
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def _compute_pr_curve(
    labels: list[int], scores: list[float],
) -> list[dict[str, float]]:
    """Compute precision-recall curve points sorted by threshold descending.

    Each point: {threshold, precision, recall}
    """
    n_pos = sum(labels)
    if n_pos == 0:
        return []

    # Sort by score descending
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    points = []
    tp = 0
    fp = 0
    prev_score = None
    for score, label in paired:
        if label:
            tp += 1
        else:
            fp += 1
        # Only record at score changes to avoid redundant points
        if prev_score is not None and score == prev_score:
            continue
        prev_score = score
        p = _precision(tp, fp)
        r = _recall(tp, n_pos - tp)
        points.append({"threshold": score, "precision": p, "recall": r})

    # Always include the (0, 1) endpoint
    if not points or points[-1]["recall"] < 1.0:
        points.append({"threshold": float("-inf"), "precision": 0.0, "recall": 1.0})

    return points


def _pr_auc_from_curve(pr_curve: list[dict]) -> float:
    """Compute PR-AUC using trapezoidal rule on recall axis."""
    if len(pr_curve) < 2:
        return 0.0
    area = 0.0
    for i in range(1, len(pr_curve)):
        r1 = pr_curve[i - 1]["recall"]
        r2 = pr_curve[i]["recall"]
        p1 = pr_curve[i - 1]["precision"]
        p2 = pr_curve[i]["precision"]
        area += (r2 - r1) * (p1 + p2) / 2
    return area


def _compute_roc_auc(labels: list[int], scores: list[float]) -> float:
    """Compute ROC-AUC using rank-based method."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    paired = sorted(zip(scores, labels), key=lambda x: x[0])
    # Rank computation with ties handled by average rank
    ranks: list[float] = []
    i = 0
    while i < len(paired):
        j = i
        while j < len(paired) and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # average rank (1-based)
        for k in range(i, j):
            ranks.append(avg_rank)
        i = j

    sum_pos_ranks = sum(r for r, (_, lbl) in zip(ranks, paired) if lbl)
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def _compute_ks(labels: list[int], scores: list[float]) -> float:
    """Compute Kolmogorov-Smirnov statistic."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    paired = sorted(zip(scores, labels), key=lambda x: x[0])
    cum_pos = 0
    cum_neg = 0
    max_ks = 0.0
    i = 0
    while i < len(paired):
        # Process all samples with the same score as a group
        j = i
        while j < len(paired) and paired[j][0] == paired[i][0]:
            if paired[j][1]:
                cum_pos += 1
            else:
                cum_neg += 1
            j += 1
        cdf_pos = cum_pos / n_pos
        cdf_neg = cum_neg / n_neg
        max_ks = max(max_ks, abs(cdf_pos - cdf_neg))
        i = j
    return max_ks


def _brier_score(labels: list[int], probs: list[float]) -> float:
    """Compute Brier score (mean squared error of probabilities)."""
    if not labels:
        return 0.0
    return sum((p - l) ** 2 for p, l in zip(probs, labels)) / len(labels)


def _calibration_error(labels: list[int], probs: list[float], n_bins: int = 10) -> float:
    """Compute expected calibration error (ECE) with equal-width bins."""
    if not labels:
        return 0.0
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = [(p > lo or i == 0) and p <= hi for p in probs]
        bin_labels = [l for l, m in zip(labels, mask) if m]
        bin_probs = [p for p, m in zip(probs, mask) if m]
        if not bin_labels:
            continue
        avg_conf = sum(bin_probs) / len(bin_probs)
        avg_acc = sum(bin_labels) / len(bin_labels)
        ece += len(bin_labels) / len(labels) * abs(avg_conf - avg_acc)
    return ece


# ── Main entry point ─────────────────────────────────────────────

def evaluate_risk(
    labels: list,
    predictions: list,
    scores: Optional[list[float]] = None,
    rewards: Optional[list[float]] = None,
    config: Optional[RiskEvalConfig] = None,
) -> RiskEvalResult:
    """Run risk control evaluation.

    Args:
        labels: Ground truth binary labels (0/1 or False/True).
        predictions: Model predictions (binary labels or probability scores).
        scores: Optional probability/score for threshold scanning.
                If None and config.has_probability, predictions are used as scores.
        rewards: Optional RL training rewards for agreement check.
        config: Evaluation configuration.

    Returns:
        RiskEvalResult with all computed metrics.
    """
    if config is None:
        config = RiskEvalConfig()

    result = RiskEvalResult(config_version=config.config_version)
    result.total_samples = len(labels)

    # Convert to binary
    pos_label = config.positive_label
    bin_labels = [1 if l == pos_label or (l is True and pos_label == 1) else 0 for l in labels]

    result.positive_count = sum(bin_labels)
    result.negative_count = result.total_samples - result.positive_count
    result.positive_rate = result.positive_count / result.total_samples if result.total_samples else 0.0

    # Degenerate check
    if result.positive_count == 0:
        result.degenerate = True
        result.degenerate_reason = "No positive samples in ground truth"
        return result
    if result.negative_count == 0:
        result.degenerate = True
        result.degenerate_reason = "No negative samples in ground truth"
        return result

    # Determine binary predictions
    if config.has_probability and scores is not None:
        # Use scores with default threshold 0.5
        bin_preds = [1 if s >= 0.5 else 0 for s in scores]
        prob_scores = scores
    elif config.has_probability and scores is None:
        # predictions are probabilities
        bin_preds = [1 if p >= 0.5 else 0 for p in predictions]
        prob_scores = [float(p) for p in predictions]
    else:
        bin_preds = [1 if p == pos_label or (p is True and pos_label == 1) else 0 for p in predictions]
        prob_scores = scores if scores else [float(p) for p in bin_preds]

    # Basic confusion matrix metrics
    tp, fp, tn, fn = _confusion_counts(bin_labels, bin_preds)
    result.precision = _precision(tp, fp)
    result.recall = _recall(tp, fn)
    result.f1 = _fbeta(result.precision, result.recall, 1.0)
    result.f2 = _fbeta(result.precision, result.recall, 2.0)
    result.false_negative_count = fn
    result.false_negative_rate = fn / result.positive_count if result.positive_count else 0.0

    # PR curve and PR-AUC
    result.pr_curve = _compute_pr_curve(bin_labels, prob_scores)
    result.pr_auc = _pr_auc_from_curve(result.pr_curve)

    # ROC-AUC
    result.roc_auc = _compute_roc_auc(bin_labels, prob_scores)

    # KS
    result.ks = _compute_ks(bin_labels, prob_scores)

    # Recall@Precision >= P
    if config.precision_floor is not None:
        best_recall = 0.0
        for pt in result.pr_curve:
            if pt["precision"] >= config.precision_floor:
                best_recall = max(best_recall, pt["recall"])
        result.recall_at_precision = best_recall

    # Precision@Recall >= R
    if config.recall_floor is not None:
        best_prec = 0.0
        for pt in result.pr_curve:
            if pt["recall"] >= config.recall_floor:
                best_prec = max(best_prec, pt["precision"])
        result.precision_at_recall = best_prec

    # Top-k recall and lift
    if config.top_k is not None and config.top_k > 0:
        k = min(config.top_k, result.total_samples)
        paired = sorted(zip(prob_scores, bin_labels), key=lambda x: -x[0])
        topk_labels = [l for _, l in paired[:k]]
        topk_pos = sum(topk_labels)
        result.topk_recall = topk_pos / result.positive_count if result.positive_count else 0.0
        base_rate = result.positive_count / result.total_samples
        if base_rate > 0:
            result.topk_lift = (topk_pos / k) / base_rate

    # Brier and calibration (only with probabilities)
    if config.has_probability:
        result.brier_score = _brier_score(bin_labels, prob_scores)
        result.calibration_error = _calibration_error(bin_labels, prob_scores)

    # Reward-eval agreement
    if rewards is not None:
        r_thresh = config.reward_threshold if config.reward_threshold is not None else 0.5
        bin_rewards = [1 if r >= r_thresh else 0 for r in rewards]
        agree = sum(1 for r, p in zip(bin_rewards, bin_preds) if r == p)
        result.reward_eval_agreement = agree / result.total_samples if result.total_samples else 0.0
        result.reward_eval_disagreement_count = result.total_samples - agree
        for i, (r, p) in enumerate(zip(bin_rewards, bin_preds)):
            if r != p:
                result.reward_eval_mismatches.append({
                    "index": i,
                    "reward": rewards[i],
                    "eval_pred": p,
                    "label": bin_labels[i],
                })

    # False negative badcases
    for i, (lbl, prd) in enumerate(zip(bin_labels, bin_preds)):
        if lbl == 1 and prd == 0:
            entry = {"index": i, "label": 1, "prediction": 0}
            if scores:
                entry["score"] = scores[i]
            if rewards:
                entry["reward"] = rewards[i]
            result.false_negative_badcases.append(entry)

    return result
