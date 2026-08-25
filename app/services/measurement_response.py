from typing import Any, Dict


DEFAULT_HEIGHT_MARGIN_CM = 0.5
DEFAULT_METHOD = "h-align-v1"
DEFAULT_KNEE_ANGLE = 180.0


def build_success_response(
    baby_id: str,
    calc_result: Dict[str, Any],
    card_result: Dict[str, Any],
    pose_result: Dict[str, Any],
    margin_cm: float = DEFAULT_HEIGHT_MARGIN_CM,
) -> Dict[str, Any]:
    height_cm = round(float(calc_result["height_cm"]), 1)
    confidence = round(
        (float(card_result["confidence"]) + float(pose_result["confidence"])) / 2,
        2,
    )

    return {
        "success": True,
        "data": {
            "baby_id": baby_id,
            "height_cm": height_cm,
            "height_range": {
                "min": round(height_cm - margin_cm, 1),
                "max": round(height_cm + margin_cm, 1),
            },
            "confidence": confidence,
            "method": DEFAULT_METHOD,
            "segments_cm": calc_result["segments"],
            "knee_angle": DEFAULT_KNEE_ANGLE,
            "warnings": [],
        },
        "error": None,
    }


def build_error_response(code: str, message: str) -> Dict[str, Any]:
    return {
        "success": False,
        "data": None,
        "error": {
            "code": code,
            "message": message,
        },
    }
