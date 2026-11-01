from pydantic import BaseModel, ConfigDict, Field, model_validator


class AsteroidFeatures(BaseModel):
    est_diameter_min: float = Field(
        ..., gt=0, description="Estimated minimum diameter, kilometers."
    )
    est_diameter_max: float = Field(
        ..., gt=0, description="Estimated maximum diameter, kilometers."
    )
    relative_velocity: float = Field(
        ..., gt=0, description="Relative velocity to Earth, km/h."
    )
    miss_distance: float = Field(
        ..., gt=0, description="Miss distance from Earth, kilometers."
    )
    absolute_magnitude: float = Field(
        ..., description="Absolute magnitude (H), a measure of intrinsic brightness."
    )

    @model_validator(mode="after")
    def check_diameter_order(self) -> "AsteroidFeatures":
        if self.est_diameter_min > self.est_diameter_max:
            raise ValueError("est_diameter_min cannot be greater than est_diameter_max.")
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "est_diameter_min": 0.1,
                "est_diameter_max": 0.25,
                "relative_velocity": 45000.0,
                "miss_distance": 5000000.0,
                "absolute_magnitude": 21.5,
            }
        }
    )


class PredictionResponse(BaseModel):
    hazardous: bool
    hazard_probability: float = Field(..., ge=0, le=1)
    model_name: str
    risk_tier: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    feature_columns: list[str]
    test_metrics: dict
    feature_importance: dict | None = None
    note: str
