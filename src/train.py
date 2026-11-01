"""
Train, tune, and evaluate the NEO Sentinel hazard classifier.

Pipeline:
    raw CSV -> clean -> train/val/test split (stratified) -> scale features
    -> baseline Logistic Regression -> candidate Random Forest and XGBoost
    (RandomizedSearchCV on the training set, model selection on the
    validation set) -> final evaluation on the held-out test set
    -> SHAP explainability -> persist pipeline + metrics + feature importance

Run:
    python src/train.py --data data/neo.csv

Outputs (written to models/):
    pipeline.joblib        StandardScaler + chosen classifier, fit on train+val
    metrics.json           baseline and final metrics, measured on this run
    feature_importance.json
    shap_summary.png       (only if shap is installed; optional, non-blocking)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import FEATURE_COLUMNS, clean, load_raw, split_features_target  # noqa: E402

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42


def evaluate(model, X, y) -> dict:
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    cm = confusion_matrix(y, preds).tolist()
    return {
        "accuracy": round(accuracy_score(y, preds), 4),
        "precision": round(precision_score(y, preds), 4),
        "recall": round(recall_score(y, preds), 4),
        "f1": round(f1_score(y, preds), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "confusion_matrix": cm,
        "confusion_matrix_labels": ["true_negative_pred_negative", "see tn,fp,fn,tp order"],
    }


def main(data_path: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path} ...")
    raw = load_raw(data_path)
    df = clean(raw)
    X, y = split_features_target(df)

    print(f"Rows after cleaning: {len(df)}")
    print(f"Class balance: {y.value_counts(normalize=True).round(4).to_dict()}")

    # 70 / 15 / 15 stratified split: train, validation, test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}  Validation: {len(X_val)}  Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    results = {}

    # --- Baseline: Logistic Regression ---
    baseline = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    baseline.fit(X_train_s, y_train)
    results["baseline_logistic_regression"] = evaluate(baseline, X_val_s, y_val)
    print("Baseline (Logistic Regression) validation metrics:", results["baseline_logistic_regression"])

    # --- Candidate 1: Random Forest, tuned ---
    rf_param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [6, 10, 16, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=rf_param_grid,
        n_iter=10,
        scoring="f1",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_search.fit(X_train_s, y_train)
    rf_best = rf_search.best_estimator_
    results["random_forest"] = evaluate(rf_best, X_val_s, y_val)
    results["random_forest"]["best_params"] = rf_search.best_params_
    print("Random Forest validation metrics:", results["random_forest"])

    candidates = {"random_forest": rf_best}

    # --- Candidate 2: XGBoost, tuned (optional if xgboost is installed) ---
    if HAS_XGBOOST:
        neg, pos = np.bincount(y_train)
        scale_pos_weight = neg / pos
        xgb_param_grid = {
            "n_estimators": [200, 300, 400],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
        }
        xgb_search = RandomizedSearchCV(
            XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            param_distributions=xgb_param_grid,
            n_iter=10,
            scoring="f1",
            cv=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        xgb_search.fit(X_train_s, y_train)
        xgb_best = xgb_search.best_estimator_
        results["xgboost"] = evaluate(xgb_best, X_val_s, y_val)
        results["xgboost"]["best_params"] = xgb_search.best_params_
        print("XGBoost validation metrics:", results["xgboost"])
        candidates["xgboost"] = xgb_best
    else:
        print("xgboost is not installed; skipping the XGBoost candidate.")

    # --- Model selection: pick the candidate with the highest validation F1 ---
    best_name = max(candidates, key=lambda name: results[name]["f1"])
    best_model = candidates[best_name]
    print(f"Selected model: {best_name} (validation F1 = {results[best_name]['f1']})")

    # Refit the chosen model and scaler on train + validation combined,
    # then report the final, unbiased estimate on the untouched test set.
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    final_scaler = StandardScaler()
    X_trainval_s = final_scaler.fit_transform(X_trainval)
    X_test_s_final = final_scaler.transform(X_test)

    best_model.fit(X_trainval_s, y_trainval)
    test_metrics = evaluate(best_model, X_test_s_final, y_test)
    print("FINAL test-set metrics (selected model, never seen during tuning):", test_metrics)

    pipeline = Pipeline([("scaler", final_scaler), ("model", best_model)])

    joblib.dump(
        {"pipeline": pipeline, "feature_columns": FEATURE_COLUMNS, "model_name": best_name},
        MODELS_DIR / "pipeline.joblib",
    )

    metrics_report = {
        "selected_model": best_name,
        "validation_metrics": {k: v for k, v in results.items()},
        "test_metrics": test_metrics,
        "dataset_rows_after_cleaning": len(df),
        "class_balance": y.value_counts(normalize=True).round(4).to_dict(),
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_report, f, indent=2, default=str)

    # Feature importance (model-native; works for both RF and XGBoost)
    importances = getattr(best_model, "feature_importances_", None)
    if importances is not None:
        importance_report = dict(
            sorted(
                zip(FEATURE_COLUMNS, [round(float(v), 4) for v in importances]),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )
        with open(MODELS_DIR / "feature_importance.json", "w") as f:
            json.dump(importance_report, f, indent=2)
        print("Feature importance:", importance_report)

    # SHAP explainability summary (best-effort; never blocks training)
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        explainer = shap.TreeExplainer(best_model)
        sample = X_test.sample(min(500, len(X_test)), random_state=RANDOM_STATE)
        sample_scaled = final_scaler.transform(sample)
        shap_values = explainer.shap_values(sample_scaled)
        shap.summary_plot(shap_values, sample, show=False)
        plt.tight_layout()
        plt.savefig(MODELS_DIR / "shap_summary.png", dpi=150)
        plt.close()
        print("SHAP summary plot saved to models/shap_summary.png")
    except Exception as exc:  # pragma: no cover - explainability is a bonus, not required
        print(f"Skipped SHAP summary plot ({exc}).")

    print("\nTraining complete.")
    print(f"Pipeline saved to: {MODELS_DIR / 'pipeline.joblib'}")
    print(f"Metrics saved to: {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/neo.csv", help="Path to the raw NEO CSV file.")
    args = parser.parse_args()
    main(args.data)
