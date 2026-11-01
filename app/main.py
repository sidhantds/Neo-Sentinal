"""
NEO Sentinel API

Serves hazard predictions for near Earth objects from the pipeline trained
by src/train.py. Loads one saved pipeline (scaler + classifier) at startup
and reuses it for every request, so prediction-time preprocessing always
matches training-time preprocessing exactly.
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from preprocess import DataValidationError, validate_prediction_input  # noqa: E402

from app.schemas import (  # noqa: E402
    AsteroidFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("neo-sentinel")

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
PIPELINE_PATH = MODELS_DIR / "pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"

_state = {"pipeline": None, "feature_columns": None, "model_name": None, "metrics": None, "importance": None}


def load_artifacts() -> None:
    if not PIPELINE_PATH.exists():
        logger.warning(
            "No trained pipeline found at %s. Run 'python src/train.py' first. "
            "The API will start but /predict will return 503 until a pipeline exists.",
            PIPELINE_PATH,
        )
        return

    bundle = joblib.load(PIPELINE_PATH)
    _state["pipeline"] = bundle["pipeline"]
    _state["feature_columns"] = bundle["feature_columns"]
    _state["model_name"] = bundle["model_name"]
    logger.info("Loaded pipeline: %s", _state["model_name"])

    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            _state["metrics"] = json.load(f)

    if IMPORTANCE_PATH.exists():
        with open(IMPORTANCE_PATH) as f:
            _state["importance"] = json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="NEO Sentinel API",
    description=(
        "Predicts whether a near Earth object is classified as hazardous, "
        "based on its size, velocity, miss distance, and brightness. "
        "Trained on NASA's Nearest Earth Objects dataset (NeoWs)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _risk_tier(probability: float) -> str:
    if probability < 0.2:
        return "low"
    if probability < 0.5:
        return "moderate"
    if probability < 0.8:
        return "elevated"
    return "high"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_state["pipeline"] is not None)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: AsteroidFeatures) -> PredictionResponse:
    if _state["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model is loaded. Run 'python src/train.py' to produce models/pipeline.joblib.",
        )

    record = features.model_dump()

    try:
        validate_prediction_input(record)
    except DataValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        row = pd.DataFrame([record], columns=_state["feature_columns"])
        proba = float(_state["pipeline"].predict_proba(row)[0, 1])
        is_hazardous = bool(proba >= 0.5)
    except Exception as exc:  # defensive: never leak a raw stack trace to the client
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PredictionResponse(
        hazardous=is_hazardous,
        hazard_probability=round(proba, 4),
        model_name=_state["model_name"],
        risk_tier=_risk_tier(proba),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if _state["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model is loaded. Run 'python src/train.py' to produce models/pipeline.joblib.",
        )

    return ModelInfoResponse(
        model_name=_state["model_name"],
        feature_columns=_state["feature_columns"],
        test_metrics=(_state["metrics"] or {}).get("test_metrics", {}),
        feature_importance=_state["importance"],
        note=(
            "Metrics are measured on a held out test set from the run that produced "
            "this pipeline file. Re-run src/train.py to regenerate them."
        ),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # pragma: no cover
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
