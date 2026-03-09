import cv2
import numpy as np
from ultralytics import YOLO
import os


class CardDetector:
    def __init__(self, model_path: str = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "yolo_card_detector.pt")

        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)

        # [수정 1] 기준 객체를 '신용카드'에서 '핸드폰(일반적인 크기)'으로 변경
        # 나중에 카드 학습 후에는 다시 8.56으로 돌려야 합니다.
        # 일반적인 스마트폰 세로 길이 (약 15cm 가정)
        self.CARD_WIDTH_CM = 15.0
        self.CARD_HEIGHT_CM = 7.2

    def detect(self, img: np.ndarray) -> dict:
        """
        이미지에서 '핸드폰'을 찾아 px_per_cm를 반환 (테스트용)
        """
        # [수정 2] classes=[67] 추가
        # 0: 사람, 67: 핸드폰 (COCO 데이터셋 기준)
        # 이렇게 하면 사람이 있어도 무시하고 핸드폰만 찾습니다.
        results = self.model(img, conf=0.3, classes=[67], verbose=False)

        if not results or len(results[0].boxes) == 0:
            return {"detected": False, "px_per_cm": 0.0, "confidence": 0.0, "bbox": []}

        # 가장 신뢰도 높은 객체 선택
        box = results[0].boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())

        w_px = x2 - x1
        h_px = y2 - y1

        # 긴 쪽을 15cm(핸드폰 길이)로 가정
        long_side_px = max(w_px, h_px)

        px_per_cm = long_side_px / self.CARD_WIDTH_CM

        return {
            "detected": True,
            "px_per_cm": float(px_per_cm),
            "confidence": conf,
            "bbox": [float(x1), float(y1), float(x2), float(y2)]
        }