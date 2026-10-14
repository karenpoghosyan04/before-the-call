import numpy as np
import pytest
from sklearn.metrics import auc, f1_score, precision_recall_curve

from bankstudy.evaluate import choose_threshold, metrics


def test_threshold_maximizes_validation_f1_and_handles_ties():
    y = np.array([0, 1, 0, 1, 0, 0])
    p = np.array([0.1, 0.7, 0.2, 0.3, 0.3, 0.9])
    threshold = choose_threshold(y, p)
    candidates = np.unique(p)
    values = np.array([f1_score(y, p >= t) for t in candidates])
    expected = candidates[np.flatnonzero(np.isclose(values, values.max()))[-1]]
    assert threshold == expected


def test_constant_scores_dummy_ap_equals_prevalence():
    y = np.array([0, 0, 0, 1])
    p = np.full(4, 0.1)
    m = metrics(y, p, choose_threshold(y, p))
    assert m["average_precision"] == 0.25
    assert m["roc_auc"] == 0.5
    assert m["confusion_matrix"] == [[0, 3], [0, 1]]


def test_ap_is_not_trapezoidal_pr_auc():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    precision, recall, _ = precision_recall_curve(y, p)
    assert not np.isclose(metrics(y, p, 0.5)["average_precision"], auc(recall, precision))


def test_malformed_predictions_are_rejected():
    with pytest.raises(ValueError):
        metrics([0, 1], [0.2, np.nan], 0.5)
    with pytest.raises(ValueError):
        metrics([0, 1], [0.2, 1.2], 0.5)


def test_saved_full_run_is_consistent_when_available():
    from pathlib import Path
    from bankstudy.report import verify
    run = Path(__file__).resolve().parents[1] / "results/full"
    if not run.exists():
        pytest.skip("Full experiment has not been executed")
    verify(run)
