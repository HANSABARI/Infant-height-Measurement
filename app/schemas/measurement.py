from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HeightRange(BaseModel):
    min: float
    max: float


class MeasurementError(BaseModel):
    code: str
    message: str


class MeasurementResult(BaseModel):
    estimatedHeightCm: float
    heightRangeCm: HeightRange | None
    confidence: float | None
    quality: Literal["GOOD", "FAIR"] | None
    modelVersion: str
    warnings: list[str] = Field(default_factory=list)
    measuredAt: datetime


class MeasurementResponse(BaseModel):
    measurementId: str
    status: Literal["SUCCESS", "RETRY", "FAILED"]
    result: MeasurementResult | None
    error: MeasurementError | None
