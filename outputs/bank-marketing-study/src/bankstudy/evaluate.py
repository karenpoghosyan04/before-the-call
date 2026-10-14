"""Threshold selection and metrics; AP is NOT trapezoidal PR-AUC."""

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)


def choose_threshold(y_val, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_val, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    # Among equal maximum F1 values, prefer the higher threshold (fewer calls).
    index = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[index])


def metrics(y, probabilities, threshold):
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    if y.shape != probabilities.shape or not np.isfinite(probabilities).all():
        raise ValueError("Predictions must be finite and match the labels")
    if ((probabilities < 0) | (probabilities > 1)).any() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Need probabilities in [0,1] and both binary classes")
    predicted = probabilities >= threshold
    return {"average_precision": float(average_precision_score(y, probabilities)),
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
            "threshold": float(threshold), "n": len(y), "prevalence": float(y.mean())}
