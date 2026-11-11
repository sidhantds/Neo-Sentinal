"""Experiment: does an asteroid's approach history help?

NASA's hazard label depends on the minimum orbit intersection distance
(MOID), which this dataset does not have. The closest approach across all
of an asteroid's recorded passes is a much better stand in for MOID than
any single pass. This experiment measures how much that would add.

It is not the shipped model, because it changes the input contract: the
caller would need the asteroid's full approach history, not one
observation. It is the clearest next step, and these numbers justify it.

Run: python -m neo_sentinel experiment
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from neo_sentinel.config import DEFAULT_DATA_PATH, FEATURES, GROUP, MODELS_DIR, RANDOM_STATE, TARGET
from neo_sentinel.data import load_training_data
from neo_sentinel.metrics import choose_threshold, evaluate
from neo_sentinel.splitting import group_split

log = logging.getLogger(__name__)

HISTORY_FEATURES = ["closest_miss_distance", "approaches_recorded", "fastest_velocity"]


def run(
    data_path: str | Path = DEFAULT_DATA_PATH,
    out_path: str | Path = MODELS_DIR / "experiment_history_features.json",
    target_recall: float = 0.95,
) -> dict:
    df, _ = load_training_data(data_path)
    by_object = df.groupby(GROUP)
    df["closest_miss_distance"] = by_object["miss_distance"].transform("min")
    df["approaches_recorded"] = by_object["miss_distance"].transform("size")
    df["fastest_velocity"] = by_object["relative_velocity"].transform("max")

    y, groups = df[TARGET].to_numpy(), df[GROUP].to_numpy()
    tr, va, te = group_split(y, groups)

    results = {}
    for label, cols in {"single_approach": FEATURES, "with_history": FEATURES + HISTORY_FEATURES}.items():
        X = df[cols].to_numpy(dtype=float)
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, early_stopping=False, random_state=RANDOM_STATE
        ).fit(X[tr], y[tr])
        thr = choose_threshold(y[va], model.predict_proba(X[va])[:, 1], target_recall)
        trva = np.concatenate([tr, va])
        model.fit(X[trva], y[trva])
        results[label] = {"features": cols, "test": evaluate(y[te], model.predict_proba(X[te])[:, 1], thr)}
        t = results[label]["test"]
        log.info(
            "%s: AP %.4f, recall %.3f, flags %.1f%%",
            label,
            t["average_precision"],
            t["recall"],
            100 * t["flagged_share"],
        )

    report = {
        "question": "How much does an asteroid's close approach history add to a single approach?",
        "note": (
            "approaches_recorded depends on the time window the dataset covers, so a production "
            "version should compute history features from NASA's full close approach record."
        ),
        "target_recall": target_recall,
        "results": results,
    }
    out = Path(out_path)
    out.write_text(json.dumps(report, indent=2) + "\n")
    log.info("Saved %s", out)
    return report


def run_leakage(
    data_path: str | Path = DEFAULT_DATA_PATH,
    out_path: str | Path = MODELS_DIR / "experiment_split_leakage.json",
) -> dict:
    """How much does a random ROW split overstate the score?

    Trains the same model twice: once on a random row split (what most
    notebooks on this dataset do) and once on the asteroid grouped split
    used here. The row split lets the model see test asteroids during
    training, so its score is inflated.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    df, _ = load_training_data(data_path)
    X, y, groups = df[FEATURES].to_numpy(dtype=float), df[TARGET].to_numpy(), df[GROUP].to_numpy()
    idx = np.arange(len(y))

    row_tr, row_te = train_test_split(idx, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
    tr, va, te = group_split(y, groups)
    grp_tr = np.concatenate([tr, va])

    results = {}
    # Fully grown trees memorise individual asteroids, which is exactly what the leak rewards.
    for split, (a, b) in {"random_rows": (row_tr, row_te), "by_asteroid": (grp_tr, te)}.items():
        model = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=RANDOM_STATE)
        p = model.fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        results[split] = {
            "test_asteroids_also_in_training": round(float(np.isin(groups[b], groups[a]).mean()), 4),
            "average_precision": round(float(average_precision_score(y[b], p)), 4),
            "roc_auc": round(float(roc_auc_score(y[b], p)), 4),
        }
        log.info("%s: %s", split, results[split])

    report = {
        "question": "How much does splitting rows at random (instead of by asteroid) inflate the test score?",
        "model": "RandomForestClassifier(n_estimators=150), fully grown trees",
        "results": results,
    }
    Path(out_path).write_text(json.dumps(report, indent=2) + "\n")
    return report
