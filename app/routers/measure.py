import hmac
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Header, HTTPException, status
import cv2
import numpy as np
import os
from datetime import datetime

# 서비스 모듈 (우리가 만든 AI 부품들)
from app.services.card_detector_has import HaSCardDetector
from app.services.head_top_pose_estimator import HeadTopPoseEstimator
from app.services.height_calculator import HeightCalculator
from app.services.visualizer import Visualizer
from app.services.jaram_height_web_response import (
    build_failed_response as build_web_failed_response,
    build_retry_response as build_web_retry_response,
    build_success_response as build_web_success_response,
)

from app.schemas.jaram_height_web import WebMeasurementResponse

router = APIRouter()
logger = logging.getLogger(__name__)

AI_API_KEY_ENV = "H_ALIGN_AI_API_KEY"
DEBUG_API_ENABLED_ENV = "H_ALIGN_ENABLE_DEBUG_API"
DEBUG_DIR_ENV = "H_ALIGN_DEBUG_DIR"

CARD_NOT_FOUND_MESSAGE = "참조 카드를 찾지 못했습니다. 카드가 전체 보이도록 다시 촬영해주세요."
INVALID_IMAGE_MESSAGE = "이미지 파일을 읽지 못했습니다. 다른 사진으로 다시 촬영해주세요."
PERSON_NOT_FOUND_MESSAGE = "사람을 찾지 못했습니다. 전신이 나오도록 다시 촬영해주세요."
BODY_CROPPED_MESSAGE = "신체 일부가 잘렸습니다. 전신이 나오도록 다시 촬영해주세요."
BAD_POSE_MESSAGE = "신장 측정에 필요한 자세를 찾지 못했습니다. 바르게 누운 자세로 다시 촬영해주세요."
LOW_CONFIDENCE_MESSAGE = "측정 신뢰도가 낮습니다. 조명과 초점을 확인해 다시 촬영해주세요."

# --- [초기화] AI 모델 로딩 (서버 시작 시 1회만 실행됨) ---
print("--- 🚀 AI 모델 로딩 시작 ---")
card_detector = HaSCardDetector()  # 카드 찾는 놈
pose_estimator = HeadTopPoseEstimator()  # 기존 WholeBody 관절 + 학습된 정수리
height_calculator = HeightCalculator()  # 계산하는 놈
visualizer = Visualizer()  # 그림 그리는 놈
print("--- ✅ AI 모델 로딩 완료 ---")

# 디버그 이미지가 저장될 폴더 설정
DEBUG_DIR = "debug_images"


# =========================================================
# 1. [Main] 실제 키 측정 API (프론트엔드 연동용)
# =========================================================
@router.post("/measure", response_model=WebMeasurementResponse)
async def measure_height(
    file: UploadFile = File(...),
    measurement_id: str = Header(..., alias="X-Measurement-Id"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    jaram-height-web worker API.
    The request contains only a normalized image and a measurement id; this server
    never receives session, baby, or user identifiers.
    """
    _verify_ai_credentials(authorization)

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return build_web_retry_response(
            measurement_id,
            code="INVALID_IMAGE",
            message=INVALID_IMAGE_MESSAGE,
        )

    try:
        card_result = card_detector.detect(img)
        if not card_result["detected"]:
            return build_web_retry_response(
                measurement_id,
                code="CARD_NOT_FOUND",
                message=CARD_NOT_FOUND_MESSAGE,
            )

        pose_result = pose_estimator.estimate(img)
        if not pose_result["detected"]:
            return _pose_failure_response(measurement_id, pose_result.get("reason"))

        calc_result = height_calculator.calculate(
            keypoints=pose_result["keypoints"],
            px_per_cm=card_result["px_per_cm"],
        )
        if "error" in calc_result:
            return build_web_retry_response(
                measurement_id,
                code="BAD_POSE",
                message=BAD_POSE_MESSAGE,
            )

        return build_web_success_response(
            measurement_id=measurement_id,
            calc_result=calc_result,
        )
    except Exception:
        logger.exception("Measurement inference failed for measurement_id=%s", measurement_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI inference is temporarily unavailable",
        )


def _pose_failure_response(measurement_id: str, reason: Optional[str]) -> dict:
    if reason == "head_top_checkpoint_missing":
        return build_web_failed_response(
            measurement_id,
            message="AI 추론 모델을 준비하지 못했습니다.",
        )
    if reason == "unexpected_head_top_shape":
        return build_web_failed_response(measurement_id)

    retry_code, retry_message = {
        "wholebody_bbox_invalid": ("BODY_CROPPED", BODY_CROPPED_MESSAGE),
        "head_top_low_confidence": ("LOW_CONFIDENCE", LOW_CONFIDENCE_MESSAGE),
        "wholebody_keypoints_missing": ("BAD_POSE", BAD_POSE_MESSAGE),
        "head_top_not_found": ("BAD_POSE", BAD_POSE_MESSAGE),
    }.get(reason, ("PERSON_NOT_FOUND", PERSON_NOT_FOUND_MESSAGE))
    return build_web_retry_response(
        measurement_id,
        code=retry_code,
        message=retry_message,
    )


def _verify_ai_credentials(authorization: Optional[str]) -> None:
    expected_api_key = os.getenv(AI_API_KEY_ENV)
    if not expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI server authentication is not configured",
        )

    scheme, _, token = (authorization or "").partition(" ")
    if (
        scheme.lower() != "bearer"
        or not token
        or not _constant_time_text_equals(token, expected_api_key)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid AI server credentials",
        )


def _constant_time_text_equals(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def _debug_api_enabled() -> bool:
    return os.getenv(DEBUG_API_ENABLED_ENV, "0").lower() in {"1", "true", "yes", "on"}


# =========================================================
# 2. [Debug] 디버깅용 API (이미지 파일 저장용)
# =========================================================
@router.post("/measure/debug", include_in_schema=False)
async def measure_debug_save(file: UploadFile = File(...)):
    """
    [Test] 분석 결과를 시각화하여 서버 폴더(debug_images)에 저장
    - 팀원 공유용 또는 모델 성능 확인용
    - 반환값: 저장된 파일 경로 및 분석 요약
    """
    if not _debug_api_enabled():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

    # 1. 이미지 읽기
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"success": False, "error": "이미지 읽기 실패"}

    # 2. AI 추론 (카드 & 포즈)
    card_result = card_detector.detect(img)
    pose_result = pose_estimator.estimate(img)
    calc_result = None
    if card_result["detected"] and pose_result["detected"]:
        calc_result = height_calculator.calculate(
            keypoints=pose_result["keypoints"],
            px_per_cm=card_result["px_per_cm"],
        )

    # 3. 시각화 (그림 그리기)
    # 원본 이미지 위에 초록 박스와 빨간 스켈레톤을 그립니다.
    debug_img = visualizer.draw_debug(img, card_result, pose_result)

    # 4. 파일 저장
    # 파일명: debug_20260212_123000.jpg 형식
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_{timestamp}.jpg"
    debug_dir = os.getenv(DEBUG_DIR_ENV, DEBUG_DIR)
    os.makedirs(debug_dir, exist_ok=True)
    save_path = os.path.join(debug_dir, filename)

    # OpenCV로 이미지 저장
    cv2.imwrite(save_path, debug_img)
    print(f"📸 디버그 이미지 저장됨: {save_path}")

    # 5. 결과 정보 반환
    return {
        "success": True,
        "message": "이미지가 서버에 저장되었습니다.",
        "file_name": filename,
        "ai_analysis": {
            "card_detected": card_result["detected"],
            "card_conf": card_result["confidence"],
            "card_class": card_result.get("class_name"),
            "card_px_per_cm": card_result.get("px_per_cm"),
            "card_geometry_method": card_result.get("geometry_method"),
            "card_bbox": card_result.get("bbox"),
            "card_model": card_result.get("model"),
            "card_model_repo": card_result.get("model_repo"),
            "card_imgsz": getattr(card_detector, "imgsz", None),
            "card_detection_source": card_result.get("detection_source"),
            "card_tile_region": card_result.get("tile_region"),
            "person_detected": pose_result["detected"],
            "person_conf": pose_result["confidence"],
            "person_reason": pose_result.get("reason"),
            "person_model": pose_result.get("model"),
            "person_bbox": pose_result.get("bbox"),
            "head_top": (
                np.asarray(pose_result["keypoints"]["head_top"]).tolist()
                if pose_result["detected"] and "head_top" in pose_result["keypoints"]
                else None
            ),
            "head_top_conf": pose_result.get("keypoint_scores", {}).get("head_top"),
            "height_cm": (
                calc_result.get("height_cm")
                if calc_result is not None and "error" not in calc_result
                else None
            ),
            "height_segments_cm": (
                calc_result.get("segments")
                if calc_result is not None and "error" not in calc_result
                else None
            ),
            "height_calculation_error": (
                calc_result.get("error") if calc_result is not None else None
            ),
        }
    }
