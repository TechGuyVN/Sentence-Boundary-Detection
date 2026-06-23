"""Compute classification metrics for SBD evaluation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(eval_pred) -> dict:
    """HuggingFace Trainer-compatible metric function."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro")),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_complete": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "f1_incomplete": float(f1_score(labels, preds, pos_label=0, zero_division=0)),
    }


def full_report(labels: list[int], preds: list[int]) -> str:
    """Full classification report + confusion matrix."""
    report = classification_report(
        labels, preds, target_names=["incomplete", "complete"], digits=4
    )
    cm = confusion_matrix(labels, preds)
    return f"{report}\nConfusion matrix:\n{cm}"
