"""Request and response contracts for the API (also used by the app's form)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE = {"absolute_magnitude": 21.5, "relative_velocity": 45000.0, "miss_distance": 5_000_000.0}


class CloseApproach(BaseModel):
    """One observed close approach of one asteroid."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE})

    absolute_magnitude: float = Field(
        ..., gt=0, le=40, description="Absolute magnitude H. Lower means bigger and brighter."
    )
    relative_velocity: float = Field(..., gt=0, le=1_000_000, description="Speed relative to Earth, km/h.")
    miss_distance: float = Field(..., gt=0, le=1e10, description="Closest distance to Earth on this pass, km.")


class BatchRequest(BaseModel):
    objects: list[CloseApproach] = Field(..., min_length=1, max_length=1000)


class DiameterRange(BaseModel):
    min: float
    max: float


class Prediction(BaseModel):
    hazard_probability: float = Field(..., ge=0, le=1, description="Calibrated chance the object is a PHA.")
    flagged: bool = Field(..., description="True if probability is at or above the review threshold.")
    risk_tier: str
    advice: str
    estimated_diameter_km: DiameterRange
    brightness_rule_flag: bool = Field(..., description="H <= 22, the brightness half of NASA's rule.")
    warnings: list[str] = []


class PredictionResponse(Prediction):
    model_version: str
    threshold: float


class BatchSummary(BaseModel):
    total: int
    flagged: int
    tiers: dict[str, int]


class BatchResponse(BaseModel):
    model_version: str
    threshold: float
    summary: BatchSummary
    results: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
