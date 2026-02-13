from pydantic import BaseModel
from typing import Optional, List, Dict

class HeightRange(BaseModel):
    min: float
    max: float

class MeasurementResult(BaseModel):
    height_range: HeightRange
    confidence: float
    method: str
    segments_cm: Dict[str, float]
    knee_angle: float
    warnings: List[str]

class MeasurementResponse(BaseModel):
    success: bool
    result: Optional[MeasurementResult] = None
    error: Optional[str] = None