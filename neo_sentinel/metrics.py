"""Evaluation and threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _r(x: float) -> float:
    return round(float(x), 4)


def choose_threshold(y_true, proba, target_recall: float) -> float:
    """Highest threshold whose recall is still at least target_recall.

    Highest, because a higher threshold flags fewer objects, so analysts get
    the smallest review list that still meets the recall goal.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = np.where(recall[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return float(thresholds.min())
    return float(thresholds[ok.max()])


def evaluate(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": _r(threshold),
        "average_precision": _r(average_precision_score(y_true, proba)),
        "roc_auc": _r(roc_auc_score(y_true, proba)),
        "brier": _r(brier_score_loss(y_true, proba)),
        "recall": _r(recall_score(y_true, pred, zero_division=0)),
        "precision": _r(precision_score(y_true, pred, zero_division=0)),
        "f1": _r(f1_score(y_true, pred, zero_division=0)),
        # The operational number: what share of all objects lands on the review list.
        "flagged_share": _r(pred.mean()),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


PHYSICS_TOLERANCE = 0.01  # a probability drop smaller than this is ignored as noise
MAX_PHYSICS_VIOLATIONS = 0.01  # share of objects allowed to break the check


def physics_violation_rate(model, X, features) -> float:
    """Share of objects whose risk goes DOWN when made bigger or closer.

    Each object is nudged two ways: 0.5 magnitudes brighter (about 26%
    larger) and a 20% closer pass. Neither change should ever lower the
    chance that NASA lists it as hazardous. A model that often says
    otherwise is fitting noise, whatever its scores.
    """
    X = np.asarray(X, dtype=float)
    base = model.predict_proba(X)[:, 1]
    bigger = X.copy()
    bigger[:, features.index("absolute_magnitude")] -= 0.5
    closer = X.copy()
    closer[:, features.index("miss_distance")] *= 0.8
    violated = np.zeros(len(X), dtype=bool)
    for nudged in (bigger, closer):
        violated |= model.predict_proba(nudged)[:, 1] < base - PHYSICS_TOLERANCE
    return float(violated.mean())


def calibration_table(y_true, proba, bins: int = 10) -> list[dict]:
    """Predicted vs observed hazardous rate in equal size probability bins."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    order = np.argsort(proba)
    rows = []
    for chunk in np.array_split(order, bins):
        if len(chunk) == 0:
            continue
        rows.append(
            {
                "mean_predicted": _r(proba[chunk].mean()),
                "observed_rate": _r(y_true[chunk].mean()),
                "count": int(len(chunk)),
            }
        )
    return rows
