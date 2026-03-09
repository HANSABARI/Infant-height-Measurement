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


        # 카드 길이
        self.CARD_WIDTH_CM = 8.56
        self.CARD_HEIGHT_CM = 5.4

    def detect(self, img: np.ndarray) -> dict:
        # 0: person, 67: cell_phone (COCO 데이터셋 기준)
        results = self.model(img, conf=0.3, classes=[0], verbose=False)

        if not results or len(results[0].boxes) == 0:
            return {"detected": False, "px_per_cm": 0.0, "confidence": 0.0, "bbox": []}

        # 가장 신뢰도 높은 객체 선택
        box = results[0].boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())

        w_px = x2 - x1
        h_px = y2 - y1

        # 긴 쪽을 8.56cm라고 가정
        long_side_px = max(w_px, h_px)
        px_per_cm = long_side_px / self.CARD_WIDTH_CM

        return {
            "detected": True,
            "px_per_cm": float(px_per_cm),
            "confidence": conf,
            "bbox": [float(x1), float(y1), float(x2), float(y2)]
        }