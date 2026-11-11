"""NEO Sentinel API.

Run:  uvicorn neo_sentinel.api.app:app   (or: python -m neo_sentinel serve)

Settings (environment variables):
  NEO_MODEL_PATH       model bundle to load (default models/neo_sentinel.joblib)
  NEO_PREDICTION_LOG   if set, append every scored request to this JSONL file
  NEO_DRIFT_WINDOW     how many recent inputs the drift check keeps (default 5000)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from neo_sentinel import __version__
from neo_sentinel.api.schemas import (
    BatchRequest,
    BatchResponse,
    BatchSummary,
    CloseApproach,
    DiameterRange,
    HealthResponse,
    Prediction,
    PredictionResponse,
)
from neo_sentinel.bundle import TIER_ADVICE, TIERS, ModelBundle
from neo_sentinel.config import model_path
from neo_sentinel.monitoring import DriftMonitor, out_of_range

log = logging.getLogger("neo_sentinel.api")


class State:
    bundle: ModelBundle | None = None
    monitor: DriftMonitor | None = None
    log_path: Path | None = None
    log_lock = threading.Lock()


state = State()


def load(path: Path | None = None) -> None:
    path = Path(path or model_path())
    state.bundle, state.monitor = None, None
    if not path.exists():
        log.warning("No model at %s. Run `python -m neo_sentinel train`. /predict returns 503 until then.", path)
        return
    state.bundle = ModelBundle.load(path)
    state.monitor = DriftMonitor(state.bundle.reference, window=int(os.environ.get("NEO_DRIFT_WINDOW", 5000)))
    log_path = os.environ.get("NEO_PREDICTION_LOG")
    state.log_path = Path(log_path) if log_path else None
    log.info("Loaded %s (version %s)", state.bundle.model_name, state.bundle.version)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load()
    yield


app = FastAPI(
    title="NEO Sentinel API",
    version=__version__,
    description=(
        "Screens near Earth object close approaches for hazard risk. Returns a calibrated "
        "probability that NASA lists the object as potentially hazardous, and whether it "
        "should go on the review list. A triage aid, not an orbital hazard assessment."
    ),
    lifespan=lifespan,
)


def _bundle() -> ModelBundle:
    if state.bundle is None:
        raise HTTPException(503, "No trained model is loaded. Run `python -m neo_sentinel train`.")
    return state.bundle


def _score(objects: list[CloseApproach]) -> list[Prediction]:
    bundle = _bundle()
    records = [o.model_dump() for o in objects]
    scored = bundle.score(pd.DataFrame.from_records(records))
    results = [
        Prediction(
            hazard_probability=float(row.hazard_probability),
            flagged=bool(row.flagged),
            risk_tier=row.risk_tier,
            advice=TIER_ADVICE[row.risk_tier],
            estimated_diameter_km=DiameterRange(min=row.est_diameter_min_km, max=row.est_diameter_max_km),
            brightness_rule_flag=bool(row.brightness_rule_flag),
            warnings=out_of_range(rec, bundle.reference),
        )
        for rec, row in zip(records, scored.itertuples(), strict=True)
    ]
    state.monitor.add(records)
    if state.log_path:
        _append_log(records, results)
    return results


def _append_log(records: list[dict], results: list[Prediction]) -> None:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        json.dumps(
            {
                "ts": now,
                "model_version": state.bundle.version,
                **rec,
                "hazard_probability": res.hazard_probability,
                "flagged": res.flagged,
            }
        )
        for rec, res in zip(records, results, strict=True)
    ]
    with state.log_lock:
        state.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state.log_path, "a") as f:
            f.write("\n".join(lines) + "\n")


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
    if state.bundle is not None:
        response.headers["X-Model-Version"] = state.bundle.version
    return response


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Readiness probe. model_loaded is false until a trained bundle is found."""
    b = state.bundle
    return HealthResponse(status="ok", model_loaded=b is not None, model_version=b.version if b else None)


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(approach: CloseApproach) -> PredictionResponse:
    """Score one close approach."""
    result = _score([approach])[0]
    b = _bundle()
    return PredictionResponse(**result.model_dump(), model_version=b.version, threshold=b.threshold)


@app.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
def predict_batch(batch: BatchRequest) -> BatchResponse:
    """Score up to 1000 close approaches in one call. Results keep the input order."""
    results = _score(batch.objects)
    b = _bundle()
    tiers = {t: sum(r.risk_tier == t for r in results) for t in TIERS}
    summary = BatchSummary(total=len(results), flagged=sum(r.flagged for r in results), tiers=tiers)
    return BatchResponse(model_version=b.version, threshold=b.threshold, summary=summary, results=results)


@app.get("/model", tags=["model"])
def model_card() -> dict:
    """Model card: version, data checksum, split, every candidate's scores, test results."""
    return _bundle().card


@app.get("/monitoring/drift", tags=["model"])
def drift() -> dict:
    """Compare recent inputs with the training data (population stability index per feature)."""
    _bundle()
    return state.monitor.report()


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):  # pragma: no cover
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
