from datetime import datetime, timezone
from typing import Any, Dict, Optional


MODEL_VERSION = "h-align-has-rtmpose-head-top-v1"


def build_success_response(
    measurement_id: str,
    calc_result: Dict[str, Any],
    measured_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Convert internal inference results to jaram-height-web's public AI contract."""
    height_cm = round(float(calc_result["height_cm"]), 1)
    measured_at = measured_at or datetime.now(timezone.utc)
    measured_at_utc = measured_at.astimezone(timezone.utc).replace(microsecond=0)

    return {
        "measurementId": measurement_id,
        "status": "SUCCESS",
        "result": {
            "estimatedHeightCm": height_cm,
            # These metrics require calibration against ground-truth data.
            "heightRangeCm": None,
            "confidence": None,
            "quality": None,
            "modelVersion": MODEL_VERSION,
            "warnings": [],
            "measuredAt": measured_at_utc.isoformat().replace("+00:00", "Z"),
        },
        "error": None,
    }


def build_retry_response(
    measurement_id: str,
    code: str,
    message: str,
) -> Dict[str, Any]:
    """Return an expected, user-recoverable inference failure."""
    return {
        "measurementId": measurement_id,
        "status": "RETRY",
        "result": None,
        "error": {
            "code": code,
            "message": message,
        },
    }


def build_failed_response(
    measurement_id: str,
    message: str = "AI 추론을 완료하지 못했습니다.",
) -> Dict[str, Any]:
    """Return a contract-valid terminal failure without exposing inference internals."""
    return {
        "measurementId": measurement_id,
        "status": "FAILED",
        "result": None,
        "error": {
            "code": "INFERENCE_FAILED",
            "message": message,
        },
    }
