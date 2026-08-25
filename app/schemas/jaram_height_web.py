from typing import List, Literal, Optional, Union

from pydantic import BaseModel


class WebHeightRange(BaseModel):
    min: float
    max: float


class WebMeasurementResult(BaseModel):
    estimatedHeightCm: float
    heightRangeCm: Optional[WebHeightRange]
    confidence: Optional[float]
    quality: Optional[Literal["GOOD", "FAIR"]]
    modelVersion: str
    warnings: List[str]
    measuredAt: str


class WebRetryError(BaseModel):
    code: Literal[
        "INVALID_IMAGE",
        "CARD_NOT_FOUND",
        "PERSON_NOT_FOUND",
        "BODY_CROPPED",
        "BAD_ANGLE",
        "BAD_POSE",
        "IMAGE_BLURRED",
        "LOW_CONFIDENCE",
    ]
    message: str


class WebFailedError(BaseModel):
    code: Literal["INFERENCE_FAILED"]
    message: str


class WebMeasurementSuccessResponse(BaseModel):
    measurementId: str
    status: Literal["SUCCESS"]
    result: WebMeasurementResult
    error: None = None


class WebMeasurementRetryResponse(BaseModel):
    measurementId: str
    status: Literal["RETRY"]
    result: None = None
    error: WebRetryError


class WebMeasurementFailedResponse(BaseModel):
    measurementId: str
    status: Literal["FAILED"]
    result: None = None
    error: WebFailedError


WebMeasurementResponse = Union[
    WebMeasurementSuccessResponse,
    WebMeasurementRetryResponse,
    WebMeasurementFailedResponse,
]
