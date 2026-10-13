"""Metrics and threshold selection. Spam (label 1) is the positive class throughout."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

TARGET_PRECISION = 0.98  # at most ~2% of messages flagged as spam may be real messages


def metrics_at(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "accuracy": float((tp + tn) / len(y_true)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision: threshold-independent, and honest on imbalanced data
    (always predicting 'ham' scores ~0.13 here, not the 87% accuracy would suggest)."""
    return float(average_precision_score(y_true, scores))


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, target_precision: float = TARGET_PRECISION) -> float:
    """Highest-recall threshold whose precision meets the target; if none does,
    the threshold with the best F1. Use validation data, never test data."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision/recall have one more entry than thresholds; drop the final (recall=0) point.
    precision, recall = precision[:-1], recall[:-1]
    meets = precision >= target_precision
    if meets.any():
        best = np.flatnonzero(meets)[np.argmax(recall[meets])]
    else:
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
        best = int(np.argmax(f1))
    return float(thresholds[best])
