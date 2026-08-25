from __future__ import annotations

import hmac
import logging
import math
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile

from app.schemas.measurement import (
    HeightRange,
    MeasurementError,
    MeasurementResponse,
    MeasurementResult,
)
if TYPE_CHECKING:
    import numpy as np

    from app.services.card_detector_rtm import CardDetector
    from app.services.height_calculator import HeightCalculator
    from app.services.pose_estimator import PoseEstimator
    from app.services.visualizer import Visualizer

logger = logging.getLogger(__name__)

AI_MODEL_VERSION = "h-align-v1"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png"}
UNCALIBRATED_WARNINGS = [
    "HEIGHT_RANGE_UNCALIBRATED",
    "CONFIDENCE_UNCALIBRATED",
]
MeasurementStatus = Literal["SUCCESS", "RETRY", "FAILED"]


class ModelServiceUnavailable(RuntimeError):
    pass


def require_ai_auth(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    configured_key = os.getenv("AI_API_KEY")
    if not configured_key:
        raise HTTPException(
            status_code=503,
            detail="AI service authentication is not configured.",
        )
    expected = f"Bearer {configured_key}"
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="AI service authentication failed.")


router = APIRouter()

_models: tuple[CardDetector, PoseEstimator, HeightCalculator, Visualizer] | None = None


def _get_models() -> tuple[CardDetector, PoseEstimator, HeightCalculator, Visualizer]:
    global _models
    if _models is not None:
        return _models

    try:
        from app.services.card_detector_rtm import CardDetector
        from app.services.height_calculator import HeightCalculator
        from app.services.pose_estimator import PoseEstimator
        from app.services.visualizer import Visualizer

        print("--- AI model loading started ---")
        _models = (
            CardDetector(),
            PoseEstimator(),
            HeightCalculator(),
            Visualizer(),
        )
        print("--- AI model loading completed ---")
        return _models
    except Exception as error:
        raise ModelServiceUnavailable from error


def _error(
    measurement_id: str,
    status: MeasurementStatus,
    code: str,
    message: str,
) -> MeasurementResponse:
    return MeasurementResponse(
        measurementId=measurement_id,
        status=status,
        result=None,
        error=MeasurementError(code=code, message=message),
    )


def _retry(measurement_id: str, code: str, message: str) -> MeasurementResponse:
    return _error(measurement_id, "RETRY", code, message)


def _failed(measurement_id: str) -> MeasurementResponse:
    return _error(
        measurement_id,
        "FAILED",
        "INFERENCE_FAILED",
        "추론을 완료하지 못했습니다. 잠시 후 다시 시도해 주세요.",
    )


def _is_transient_error(error: Exception) -> bool:
    message = str(error).lower()
    return isinstance(error, TimeoutError) or any(
        token in message
        for token in ("timeout", "timed out", "cuda", "gpu", "out of memory")
    )


def _decode_image(contents: bytes) -> np.ndarray | None:
    try:
        import cv2
        import numpy as np
    except Exception as error:
        raise ModelServiceUnavailable from error

    return cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)


@router.post(
    "/measure",
    response_model=MeasurementResponse,
    dependencies=[Depends(require_ai_auth)],
)
async def measure_height(
    file: UploadFile = File(...),
    measurement_id: Annotated[str, Header(alias="X-Measurement-Id")] = "",
) -> MeasurementResponse:
    if not measurement_id.strip():
        raise HTTPException(status_code=400, detail="X-Measurement-Id is required.")
    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        return _retry(
            measurement_id,
            "INVALID_IMAGE",
            "JPEG 또는 PNG 사진만 사용할 수 있습니다.",
        )

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        return _retry(
            measurement_id,
            "INVALID_IMAGE",
            "이미지 파일은 10MB 이하만 사용할 수 있습니다.",
        )

    try:
        image = _decode_image(contents)
        if image is None:
            return _retry(
                measurement_id,
                "INVALID_IMAGE",
                "이미지 파일을 읽을 수 없습니다.",
            )

        card_detector, pose_estimator, height_calculator, _ = _get_models()
        card_result = card_detector.detect(image)
        if not card_result["detected"]:
            return _retry(
                measurement_id,
                "CARD_NOT_FOUND",
                "참조 카드가 보이도록 다시 촬영해 주세요.",
            )

        pose_result = pose_estimator.estimate(image)
        if not pose_result["detected"]:
            return _retry(
                measurement_id,
                "PERSON_NOT_FOUND",
                "전신이 보이도록 다시 촬영해 주세요.",
            )

        calculation = height_calculator.calculate(
            keypoints=pose_result["keypoints"],
            px_per_cm=card_result["px_per_cm"],
        )
        if calculation.get("error"):
            return _failed(measurement_id)

        height_cm = calculation.get("height_cm")
        if not isinstance(height_cm, (int, float)) or not math.isfinite(height_cm):
            return _failed(measurement_id)

        return MeasurementResponse(
            measurementId=measurement_id,
            status="SUCCESS",
            result=MeasurementResult(
                estimatedHeightCm=float(height_cm),
                heightRangeCm=None,
                confidence=None,
                quality=None,
                modelVersion=AI_MODEL_VERSION,
                warnings=UNCALIBRATED_WARNINGS,
                measuredAt=datetime.now(timezone.utc),
            ),
            error=None,
        )
    except Exception as error:
        if isinstance(error, ModelServiceUnavailable):
            logger.error("AI model service is unavailable")
            raise HTTPException(
                status_code=503,
                detail="AI model service is temporarily unavailable.",
            ) from error
        logger.exception("AI inference failed")
        if _is_transient_error(error):
            raise HTTPException(
                status_code=503,
                detail="AI service is temporarily unavailable.",
            ) from error
        return _failed(measurement_id)


@router.post(
    "/measure/debug",
    dependencies=[Depends(require_ai_auth)],
)
async def measure_debug_save(file: UploadFile = File(...)) -> dict[str, object]:
    if os.getenv("DEBUG_IMAGES_ENABLED", "false").lower() != "true":
        raise HTTPException(status_code=404, detail="Not found.")

    debug_dir = os.getenv("DEBUG_IMAGE_DIR", "debug_images")
    os.makedirs(debug_dir, exist_ok=True)
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    image = _decode_image(contents)
    if image is None:
        return {"success": False, "error": "이미지 읽기 실패"}

    card_detector, pose_estimator, _, visualizer = _get_models()
    card_result = card_detector.detect(image)
    pose_result = pose_estimator.estimate(image)
    debug_image = visualizer.draw_debug(image, card_result, pose_result)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"debug_{timestamp}.jpg"
    save_path = os.path.join(debug_dir, filename)
    import cv2

    cv2.imwrite(save_path, debug_image)

    return {
        "success": True,
        "message": "이미지가 서버에 저장되었습니다.",
        "file_name": filename,
        "saved_path": os.path.abspath(save_path),
        "ai_analysis": {
            "card_detected": card_result["detected"],
            "card_conf": card_result["confidence"],
            "person_detected": pose_result["detected"],
            "person_conf": pose_result["confidence"],
        },
    }
