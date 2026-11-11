"""Train, compare and select the NEO Sentinel model.

Steps:
  1. Load and clean the CSV, verify its checksum is recorded.
  2. Split 70 / 15 / 15 into train, validation and test, by asteroid.
  3. Tune each candidate on train with asteroid grouped cross validation.
  4. Compare every candidate, including the H <= 22 rule, on validation.
     A candidate is only eligible if it passes the physics check (making an
     object bigger or its pass closer must not lower its risk). Among those,
     the highest validation average precision wins.
  5. Pick the decision threshold on validation: the highest one that still
     catches the target share of hazardous objects.
  6. Refit the winner on train + validation and score it once on test.
  7. Save the bundle (models/neo_sentinel.joblib) and a readable model card
     (models/model_card.json).
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV

from neo_sentinel import __version__
from neo_sentinel.bundle import ModelBundle
from neo_sentinel.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_TARGET_RECALL,
    FEATURES,
    GROUP,
    RANDOM_STATE,
    TARGET,
)
from neo_sentinel.data import describe_dataset, file_sha256, load_training_data
from neo_sentinel.metrics import (
    MAX_PHYSICS_VIOLATIONS,
    calibration_table,
    choose_threshold,
    evaluate,
    physics_violation_rate,
)
from neo_sentinel.models import candidates
from neo_sentinel.monitoring import reference_profile
from neo_sentinel.splitting import group_cv, group_split

log = logging.getLogger(__name__)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _tune(name, estimator, params, X, y, groups, n_iter, cv_folds):
    if params is None:  # the rule baseline has nothing to tune
        return clone(estimator).fit(X, y), {}
    n_combos = int(np.prod([len(v) for v in params.values()]))
    search = RandomizedSearchCV(
        estimator,
        params,
        n_iter=min(n_iter, n_combos),
        scoring="average_precision",
        cv=group_cv(cv_folds),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y, groups=groups)
    log.info("  %s best CV average precision %.4f", name, search.best_score_)
    return search.best_estimator_, search.best_params_


def train(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    target_recall: float = DEFAULT_TARGET_RECALL,
    n_iter: int = 12,
    cv_folds: int = 3,
    only: list[str] | None = None,
) -> ModelBundle:
    data_path, model_path = Path(data_path), Path(model_path)

    log.info("Loading %s", data_path)
    df, cleaning = load_training_data(data_path)
    facts = describe_dataset(df)
    log.info("Cleaned: %s", facts)

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy()
    groups = df[GROUP].to_numpy()
    tr, va, te = group_split(y, groups)
    log.info("Split by asteroid: train %d, validation %d, test %d rows", len(tr), len(va), len(te))

    # ---- tune and compare on validation
    pool = candidates()
    if only:
        pool = {k: v for k, v in pool.items() if k in only or k == "brightness_rule"}

    fitted, best_params, validation = {}, {}, {}
    for name, (estimator, params) in pool.items():
        log.info("Tuning %s", name)
        model, bp = _tune(name, estimator, params, X[tr], y[tr], groups[tr], n_iter, cv_folds)
        proba = model.predict_proba(X[va])[:, 1]
        thr = choose_threshold(y[va], proba, target_recall)
        fitted[name], best_params[name] = model, bp
        validation[name] = evaluate(y[va], proba, thr)
        validation[name]["physics_violations"] = round(physics_violation_rate(model, X[va], FEATURES), 4)
        m = validation[name]
        log.info(
            "  validation: AP %.4f  recall %.3f  precision %.3f  flags %.1f%%  physics violations %.1f%%",
            m["average_precision"],
            m["recall"],
            m["precision"],
            100 * m["flagged_share"],
            100 * m["physics_violations"],
        )

    # Only models that respect the physics check may win; then highest AP.
    eligible = [n for n in validation if validation[n]["physics_violations"] <= MAX_PHYSICS_VIOLATIONS]
    if not eligible:
        log.warning("No candidate passed the physics check; selecting on scores alone.")
        eligible = list(validation)
    best = max(eligible, key=lambda n: (validation[n]["average_precision"], validation[n]["roc_auc"]))
    threshold = validation[best]["threshold"]
    log.info("Selected %s, threshold %.4f", best, threshold)

    # ---- refit on train + validation, test once
    trva = np.concatenate([tr, va])
    final = clone(fitted[best]).fit(X[trva], y[trva])
    test_proba = final.predict_proba(X[te])[:, 1]
    test_metrics = evaluate(y[te], test_proba, threshold)

    baseline = pool["brightness_rule"][0].fit(X[trva], y[trva])
    baseline_test = evaluate(y[te], baseline.predict_proba(X[te])[:, 1], 0.5)
    log.info("TEST (%s): %s", best, {k: v for k, v in test_metrics.items() if k != "confusion_matrix"})
    log.info(
        "TEST (H <= 22 rule): AP %.4f recall %.3f flags %.1f%%",
        baseline_test["average_precision"],
        baseline_test["recall"],
        100 * baseline_test["flagged_share"],
    )

    importance = permutation_importance(
        final, X[te], y[te], scoring="average_precision", n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    importance = {
        f: {"mean": round(float(m), 4), "std": round(float(s), 4)}
        for f, m, s in sorted(
            zip(FEATURES, importance.importances_mean, importance.importances_std, strict=True), key=lambda t: -t[1]
        )
    }

    # ---- metadata and card
    trained_at = datetime.now(UTC)
    data_sha = file_sha256(data_path)
    metadata = {
        "model_version": f"{trained_at:%Y.%m.%d.%H%M}-{data_sha[:8]}",
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "package_version": __version__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "git_commit": _git_commit(),
        "data_file": data_path.name,
        "data_sha256": data_sha,
    }
    bundle = ModelBundle(
        estimator=final,
        threshold=threshold,
        model_name=best,
        reference=reference_profile(pd.DataFrame(X[trva], columns=FEATURES)),
        metadata=metadata,
    )
    bundle.card = {
        "model_name": best,
        "metadata": metadata,
        "features": FEATURES,
        "target_recall": target_recall,
        "threshold": threshold,
        "tier_edges": bundle.tier_edges,
        "selection_rule": (
            f"among models with at most {MAX_PHYSICS_VIOLATIONS:.0%} physics violations on validation, "
            "highest validation average precision (ties: ROC AUC)"
        ),
        "threshold_rule": f"highest validation threshold with recall >= {target_recall}",
        "dataset": facts,
        "cleaning": cleaning.as_dict(),
        "split": {
            "method": "stratified, grouped by asteroid id (no asteroid in more than one part)",
            "rows": {"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te))},
            "asteroids": {
                "train": int(len(np.unique(groups[tr]))),
                "validation": int(len(np.unique(groups[va]))),
                "test": int(len(np.unique(groups[te]))),
            },
        },
        "validation": validation,
        "best_params": {k: {p: v for p, v in bp.items()} for k, bp in best_params.items()},
        "test": test_metrics,
        "test_baseline_brightness_rule": baseline_test,
        "calibration_test": calibration_table(y[te], test_proba),
        "permutation_importance_test": importance,
    }

    bundle.save(model_path)
    card_path = model_path.with_name("model_card.json")
    card_path.write_text(json.dumps(bundle.card, indent=2, default=str) + "\n")
    log.info("Saved %s and %s (version %s)", model_path, card_path, bundle.version)
    return bundle
