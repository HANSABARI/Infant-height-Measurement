from pydantic import BaseModel
from typing import Optional, List, Dict


class HeightRange(BaseModel):
    min: float
    max: float


class MeasurementData(BaseModel):
    baby_id: str
    height_cm: float
    height_range: HeightRange
    confidence: float
    method: str
    segments_cm: Dict[str, float]
    knee_angle: float
    warnings: List[str]


class MeasurementError(BaseModel):
    code: str
    message: str


class MeasurementResponse(BaseModel):
    success: bool
    data: Optional[MeasurementData] = None
    error: Optional[MeasurementError] = None
