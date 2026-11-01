"""
Train, tune, and evaluate the NEO Sentinel hazard classifier.

Pipeline:
    raw CSV -> clean -> stratified train / validation / test split
    -> Logistic Regression baseline, Random Forest and XGBoost candidates
       (each a full scaler + model Pipeline, tuned with RandomizedSearchCV
       so scaling is refit inside every CV fold and never leaks)
    -> pick the best candidate by validation F1
    -> refit on train + validation, score once on the untouched test set
    -> save pipeline, metrics, feature importance, optional SHAP plot

Run:
    python src/train.py --data data/neo.csv

Outputs (written to models/):
    pipeline.joblib          scaler + chosen classifier, fit on train + validation
    metrics.json             validation metrics per model and final test metrics
    feature_importance.json  model native importance (or |coef| for linear models)
    shap_summary.png         only if shap is installed; never blocks training
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import FEATURE_COLUMNS, clean, load_raw, split_features_target  # noqa: E402

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42


def evaluate(model, X, y) -> dict:
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def make_pipeline(model) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def build_searches(y_train: pd.Series, n_iter: int, cv) -> dict:
    """Each candidate is a full Pipeline, so scaling happens inside every CV fold."""
    searches = {}

    searches["logistic_regression"] = RandomizedSearchCV(
        make_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
        param_distributions={"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        n_iter=min(n_iter, 5),
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    searches["random_forest"] = RandomizedSearchCV(
        make_pipeline(RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)),
        param_distributions={
            "model__n_estimators": [200, 300, 400],
            "model__max_depth": [6, 10, 16, None],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        },
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    if HAS_XGBOOST:
        counts = np.bincount(y_train, minlength=2)
        scale_pos_weight = float(counts[0] / max(counts[1], 1))
        searches["xgboost"] = RandomizedSearchCV(
            make_pipeline(
                XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
            ),
            param_distributions={
                "model__n_estimators": [200, 300, 400],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
            },
            n_iter=n_iter,
            scoring="f1",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        print("xgboost is not installed; skipping the XGBoost candidate.")

    return searches


def feature_importance(pipeline: Pipeline) -> dict | None:
    model = pipeline.named_steps["model"]
    values = getattr(model, "feature_importances_", None)
    if values is None and hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])  # scaled inputs, so magnitudes are comparable
    if values is None:
        return None
    pairs = zip(FEATURE_COLUMNS, [round(float(v), 4) for v in values])
    return dict(sorted(pairs, key=lambda kv: kv[1], reverse=True))


def save_shap_plot(pipeline: Pipeline, X_test: pd.DataFrame, out_path: Path) -> None:
    """Best effort explainability plot. Any failure is reported and skipped."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        model = pipeline.named_steps["model"]
        scaler = pipeline.named_steps["scaler"]
        sample = X_test.sample(min(500, len(X_test)), random_state=RANDOM_STATE)
        sample_scaled = scaler.transform(sample)

        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, sample_scaled)
        shap_values = explainer.shap_values(sample_scaled)

        # Random Forest returns one array per class (list or 3D array); keep the hazardous class.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif getattr(shap_values, "ndim", 2) == 3:
            shap_values = shap_values[:, :, 1]

        # Colour by raw feature values so the plot reads in real units.
        shap.summary_plot(shap_values, sample, feature_names=FEATURE_COLUMNS, show=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"SHAP summary plot saved to {out_path}")
    except Exception as exc:  # pragma: no cover
        print(f"Skipped SHAP summary plot ({exc}).")


def main(data_path: str, models_dir: Path = DEFAULT_MODELS_DIR, n_iter: int = 10, shap_plot: bool = True) -> dict:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path} ...")
    df = clean(load_raw(data_path))
    X, y = split_features_target(df)

    if y.nunique() < 2:
        raise ValueError("The target column has only one class after cleaning; cannot train a classifier.")

    class_balance = {str(k): v for k, v in y.value_counts(normalize=True).round(4).to_dict().items()}
    print(f"Rows after cleaning: {len(df)}")
    print(f"Class balance: {class_balance}")

    # 70 / 15 / 15 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}  Validation: {len(X_val)}  Test: {len(X_test)}")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    searches = build_searches(y_train, n_iter, cv)

    results, best_params = {}, {}
    for name, search in searches.items():
        print(f"Tuning {name} ...")
        search.fit(X_train, y_train)
        results[name] = evaluate(search.best_estimator_, X_val, y_val)
        best_params[name] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        print(f"  {name} validation: F1={results[name]['f1']}  recall={results[name]['recall']}  ROC AUC={results[name]['roc_auc']}")

    # Highest validation F1 wins; ROC AUC breaks ties.
    best_name = max(results, key=lambda n: (results[n]["f1"], results[n]["roc_auc"]))
    print(f"Selected model: {best_name} (validation F1 = {results[best_name]['f1']})")

    # Refit a fresh copy with the tuned params on train + validation, then test once.
    final_pipeline = clone(searches[best_name].best_estimator_)
    final_pipeline.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    test_metrics = evaluate(final_pipeline, X_test, y_test)
    print(f"FINAL test metrics ({best_name}, never seen during tuning): {test_metrics}")

    joblib.dump(
        {"pipeline": final_pipeline, "feature_columns": FEATURE_COLUMNS, "model_name": best_name},
        models_dir / "pipeline.joblib",
    )

    metrics_report = {
        "selected_model": best_name,
        "selection_rule": "highest validation F1, ties broken by ROC AUC",
        "validation_metrics": results,
        "best_params": best_params,
        "test_metrics": test_metrics,
        "dataset_rows_after_cleaning": int(len(df)),
        "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "class_balance": class_balance,
    }
    with open(models_dir / "metrics.json", "w") as f:
        json.dump(metrics_report, f, indent=2, default=str)

    importance = feature_importance(final_pipeline)
    if importance is not None:
        with open(models_dir / "feature_importance.json", "w") as f:
            json.dump(importance, f, indent=2)
        print(f"Feature importance: {importance}")

    if shap_plot:
        save_shap_plot(final_pipeline, X_test, models_dir / "shap_summary.png")

    print("\nTraining complete.")
    print(f"Pipeline saved to: {models_dir / 'pipeline.joblib'}")
    print(f"Metrics saved to: {models_dir / 'metrics.json'}")
    return metrics_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the NEO Sentinel hazard classifier.")
    parser.add_argument("--data", default="data/neo.csv", help="Path to the raw NEO CSV file.")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR), help="Where to write model artifacts.")
    parser.add_argument("--n-iter", type=int, default=10, help="RandomizedSearchCV iterations per model.")
    parser.add_argument("--no-shap", action="store_true", help="Skip the SHAP summary plot.")
    args = parser.parse_args()
    main(args.data, Path(args.models_dir), args.n_iter, shap_plot=not args.no_shap)
