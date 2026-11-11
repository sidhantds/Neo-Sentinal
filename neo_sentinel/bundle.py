"""The model bundle: the one file that crosses from training to serving.

It holds the fitted estimator plus everything needed to use it correctly:
feature order, the decision threshold, risk tier edges, the training data
profile for drift checks, and version metadata. The API and the app both
score through ModelBundle.score(), so they cannot disagree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from neo_sentinel.config import FEATURES
from neo_sentinel.physics import brightness_rule, diameter_range_km

log = logging.getLogger(__name__)

BUNDLE_FORMAT = 2
TIERS = ["low", "moderate", "elevated", "high"]

TIER_ADVICE = {
    "low": "Below the review line. No closer look needed.",
    "moderate": "On the review list. Check when time allows.",
    "elevated": "Strong hazard signals. Recommend a closer orbital review.",
    "high": "Very likely on NASA's hazardous list. Prioritise for full orbital analysis.",
}


def tier_edges_for(threshold: float) -> list[float]:
    """Low is everything below the review threshold. Above it, the tiers
    split on probability: under 30%, 30% to 60%, and 60% or more."""
    edges = [threshold, max(threshold, 0.30), max(threshold, 0.60)]
    return [float(e) for e in edges]


@dataclass
class ModelBundle:
    estimator: object
    threshold: float
    model_name: str
    features: list = field(default_factory=lambda: list(FEATURES))
    tier_edges: list = field(default_factory=list)
    reference: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    card: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.tier_edges:
            self.tier_edges = tier_edges_for(self.threshold)

    # ------------------------------------------------------------ scoring

    @property
    def version(self) -> str:
        return self.metadata.get("model_version", "unversioned")

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        # Columns are selected by name in the stored order, so a caller that
        # sends fields in a different order cannot scramble the inputs.
        X = frame[self.features].to_numpy(dtype=float)
        return self.estimator.predict_proba(X)[:, 1]

    def tier(self, probability: float) -> str:
        return TIERS[int(np.searchsorted(self.tier_edges, probability, side="right"))]

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return frame plus hazard_probability, flagged, risk_tier and context columns."""
        proba = self.predict_proba(frame)
        out = frame.copy()
        out["hazard_probability"] = np.round(proba, 4)
        out["flagged"] = proba >= self.threshold
        out["risk_tier"] = [self.tier(p) for p in proba]
        dmin, dmax = diameter_range_km(out["absolute_magnitude"])
        out["est_diameter_min_km"] = np.round(dmin, 4)
        out["est_diameter_max_km"] = np.round(dmax, 4)
        out["brightness_rule_flag"] = brightness_rule(out["absolute_magnitude"])
        return out

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": BUNDLE_FORMAT, **self.__dict__}
        joblib.dump(payload, path, compress=3)
        return path

    @classmethod
    def load(cls, path: str | Path) -> ModelBundle:
        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"{path} is not a NEO Sentinel model bundle (format {BUNDLE_FORMAT}).")
        payload.pop("format")
        bundle = cls(**payload)
        trained_with = bundle.metadata.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            log.warning(
                "Model was trained with scikit-learn %s but %s is installed. "
                "Predictions may differ; install the pinned version from requirements.txt.",
                trained_with,
                sklearn.__version__,
            )
        return bundle
