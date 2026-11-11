"""Candidate models and their search spaces."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from neo_sentinel.config import FEATURES, RANDOM_STATE
from neo_sentinel.physics import brightness_rule


class BrightnessRule(ClassifierMixin, BaseEstimator):
    """Baseline: flag an object when H <= 22, the brightness half of NASA's rule.

    No learning. Every trained model is compared against this, because a
    model that cannot beat a one line rule is not worth deploying.
    """

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = len(FEATURES)
        return self

    def predict_proba(self, X):
        h = np.asarray(X)[:, FEATURES.index("absolute_magnitude")]
        flag = brightness_rule(h).astype(float)
        return np.column_stack([1 - flag, flag])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def candidates() -> dict:
    """name -> (estimator, param distributions for RandomizedSearchCV or None).

    Class imbalance (about 10% hazardous) is handled by choosing the decision
    threshold on validation data, not by reweighting classes. That keeps the
    predicted probabilities close to real frequencies, so "30%" means about
    three in ten such objects are on NASA's hazardous list.
    """
    logistic = Pipeline(
        [
            ("log", FunctionTransformer(np.log)),  # all inputs are positive and skewed
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )
    forest = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=RANDOM_STATE)
    # Physics we can promise: a brighter (lower H, so bigger) object, or a
    # closer pass, never makes an object LESS likely to be hazardous. Telling
    # the model that removes noisy wiggles and makes its answers explainable.
    monotonic = {"absolute_magnitude": -1, "relative_velocity": 0, "miss_distance": -1}
    boosting = HistGradientBoostingClassifier(
        monotonic_cst=[monotonic[f] for f in FEATURES], early_stopping=False, random_state=RANDOM_STATE
    )

    return {
        "brightness_rule": (BrightnessRule(), None),
        "logistic_regression": (logistic, {"model__C": [0.01, 0.1, 1.0, 10.0]}),
        "random_forest": (
            forest,
            {
                # Leaves are kept large so the saved file stays small enough to commit.
                "max_depth": [8, 12, 16],
                "min_samples_leaf": [20, 50, 100],
                "max_features": [1, 2, 3],
            },
        ),
        "hist_gradient_boosting": (
            boosting,
            {
                "learning_rate": [0.03, 0.05, 0.1],
                "max_iter": [200, 300, 500],
                "max_leaf_nodes": [15, 31, 63],
                "min_samples_leaf": [20, 50, 100],
                "l2_regularization": [0.0, 0.1, 1.0],
            },
        ),
    }
