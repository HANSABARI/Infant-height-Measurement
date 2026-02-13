from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
import os
from datetime import datetime

# 서비스 모듈 (우리가 만든 AI 부품들)
from app.services.card_detector import CardDetector
from app.services.pose_estimator import PoseEstimator
from app.services.height_calculator import HeightCalculator
from app.services.visualizer import Visualizer

# 스키마 (데이터 형틀)
from app.schemas.measurement import MeasurementResponse, MeasurementResult, HeightRange

router = APIRouter()

# --- [초기화] AI 모델 로딩 (서버 시작 시 1회만 실행됨) ---
print("--- 🚀 AI 모델 로딩 시작 ---")
card_detector = CardDetector()  # 카드 찾는 놈
pose_estimator = PoseEstimator()  # 관절 찾는 놈
height_calculator = HeightCalculator()  # 계산하는 놈
visualizer = Visualizer()  # 그림 그리는 놈
print("--- ✅ AI 모델 로딩 완료 ---")

# 디버그 이미지가 저장될 폴더 설정
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)  # 폴더 없으면 자동 생성


# =========================================================
# 1. [Main] 실제 키 측정 API (프론트엔드 연동용)
# =========================================================
@router.post("/measure", response_model=MeasurementResponse)
async def measure_height(file: UploadFile = File(...)):
    """
    [Production] 앱에서 사용하는 메인 API
    - 이미지 업로드 -> 카드/관절 검출 -> 키 계산 -> JSON 결과 반환
    """
    # 1. 이미지 읽기
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return MeasurementResponse(success=False, error="이미지 파일을 읽을 수 없습니다.")

    # 2. 카드 검출 (Stage 2)
    card_result = card_detector.detect(img)
    if not card_result["detected"]:
        return MeasurementResponse(
            success=False,
            error="참조 카드(신용카드)를 찾을 수 없습니다. 카드가 잘 보이게 찍어주세요."
        )

    # 3. 포즈 추정 (Stage 3)
    pose_result = pose_estimator.estimate(img)
    if not pose_result["detected"]:
        return MeasurementResponse(
            success=False,
            error="사람을 찾을 수 없습니다. 전신이 나오도록 찍어주세요."
        )

    # 4. 키 산출 (Stage 4)
    calc_result = height_calculator.calculate(
        keypoints=pose_result["keypoints"],
        px_per_cm=card_result["px_per_cm"]
    )

    # 5. 결과 반환
    height_cm = calc_result["height_cm"]
    margin = 0.5  # 오차 범위 (±0.5cm)

    return MeasurementResponse(
        success=True,
        result=MeasurementResult(
            height_range=HeightRange(
                min=round(height_cm - margin, 1),
                max=round(height_cm + margin, 1)
            ),
            confidence=round((card_result["confidence"] + pose_result["confidence"]) / 2, 2),
            method="h-align-v1",
            segments_cm=calc_result["segments"],
            knee_angle=180.0,  # 추후 구현 예정
            warnings=[]
        )
    )


# =========================================================
# 2. [Debug] 디버깅용 API (이미지 파일 저장용)
# =========================================================
@router.post("/measure/debug")
async def measure_debug_save(file: UploadFile = File(...)):
    """
    [Test] 분석 결과를 시각화하여 서버 폴더(debug_images)에 저장
    - 팀원 공유용 또는 모델 성능 확인용
    - 반환값: 저장된 파일 경로 및 분석 요약
    """
    # 1. 이미지 읽기
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"success": False, "error": "이미지 읽기 실패"}

    # 2. AI 추론 (카드 & 포즈)
    card_result = card_detector.detect(img)
    pose_result = pose_estimator.estimate(img)

    # 3. 시각화 (그림 그리기)
    # 원본 이미지 위에 초록 박스와 빨간 스켈레톤을 그립니다.
    debug_img = visualizer.draw_debug(img, card_result, pose_result)

    # 4. 파일 저장
    # 파일명: debug_20260212_123000.jpg 형식
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_{timestamp}.jpg"
    save_path = os.path.join(DEBUG_DIR, filename)

    # OpenCV로 이미지 저장
    cv2.imwrite(save_path, debug_img)
    print(f"📸 디버그 이미지 저장됨: {save_path}")

    # 5. 결과 정보 반환
    return {
        "success": True,
        "message": "이미지가 서버에 저장되었습니다.",
        "file_name": filename,
        "saved_path": os.path.abspath(save_path),
        "ai_analysis": {
            "card_detected": card_result["detected"],
            "card_conf": card_result["confidence"],
            "person_detected": pose_result["detected"],
            "person_conf": pose_result["confidence"]
        }
    }