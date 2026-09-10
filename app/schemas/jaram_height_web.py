from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict


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
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "measurementId": "7d474c14-6208-4380-8c7f-e9609da83f98",
                    "status": "SUCCESS",
                    "result": {
                        "estimatedHeightCm": 64.2,
                        "heightRangeCm": None,
                        "confidence": None,
                        "quality": None,
                        "modelVersion": "h-align-has-rtmpose-head-top-v1",
                        "warnings": [],
                        "measuredAt": "2026-08-23T12:30:00Z",
                    },
                    "error": None,
                }
            ]
        }
    )

    measurementId: str
    status: Literal["SUCCESS"]
    result: WebMeasurementResult
    error: None = None


class WebMeasurementRetryResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "measurementId": "7d474c14-6208-4380-8c7f-e9609da83f98",
                    "status": "RETRY",
                    "result": None,
                    "error": {
                        "code": "CARD_NOT_FOUND",
                        "message": "참조 카드를 찾지 못했습니다. 카드가 전체 보이도록 다시 촬영해주세요.",
                    },
                }
            ]
        }
    )

    measurementId: str
    status: Literal["RETRY"]
    result: None = None
    error: WebRetryError


class WebMeasurementFailedResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "measurementId": "7d474c14-6208-4380-8c7f-e9609da83f98",
                    "status": "FAILED",
                    "result": None,
                    "error": {
                        "code": "INFERENCE_FAILED",
                        "message": "AI 추론 모델을 준비하지 못했습니다.",
                    },
                }
            ]
        }
    )

    measurementId: str
    status: Literal["FAILED"]
    result: None = None
    error: WebFailedError


WebMeasurementResponse = Union[
    WebMeasurementSuccessResponse,
    WebMeasurementRetryResponse,
    WebMeasurementFailedResponse,
]
