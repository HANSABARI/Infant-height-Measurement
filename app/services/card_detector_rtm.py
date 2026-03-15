import cv2
import numpy as np
import os
import torch
from mmdet.apis import init_detector, inference_detector

# Monkey Patching(weights_only=False)
# =====================================================================
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
# =====================================================================

class CardDetector:
    def __init__(self, config_path: str = None, checkpoint_path: str = None, device: str = 'cpu'):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if config_path is None:
            config_path = os.path.join(base_dir, "models", "card_model_v2", "rtmdet_nano_card.py")
        if checkpoint_path is None:
            checkpoint_path = os.path.join(base_dir, "models", "card_model_v2", "epoch_50.pth")

        print(f"Loading RTMDet model...\nConfig: {config_path}\nWeights: {checkpoint_path}")
        self.model = init_detector(config_path, checkpoint_path, device=device)

        # 카드 길이 (표준 신용카드)
        self.CARD_WIDTH_CM = 8.56
        self.CARD_HEIGHT_CM = 5.4

    def detect(self, img: np.ndarray) -> dict:
        # 1. RTMDet 추론 실행
        result = inference_detector(self.model, img)

        # 2. 파싱
        pred_instances = result.pred_instances
        scores = pred_instances.scores.cpu().numpy()
        bboxes = pred_instances.bboxes.cpu().numpy()

        # 3. 신뢰도 0.3 이상 필터링
        valid_indices = scores > 0.3

        if not valid_indices.any():
            return {"detected": False, "px_per_cm": 0.0, "confidence": 0.0, "bbox": []}

        # 4. 가장 신뢰도 높은 객체 선택
        best_idx = scores[valid_indices].argmax()
        valid_bboxes = bboxes[valid_indices]
        valid_scores = scores[valid_indices]

        x1, y1, x2, y2 = valid_bboxes[best_idx]
        conf = float(valid_scores[best_idx])

        w_px = x2 - x1
        h_px = y2 - y1

        # 긴 쪽을 8.56cm라고 가정하여 비율 계산
        long_side_px = max(w_px, h_px)
        px_per_cm = float(long_side_px / self.CARD_WIDTH_CM)

        return {
            "detected": True,
            "px_per_cm": px_per_cm,
            "confidence": conf,
            "bbox": [float(x1), float(y1), float(x2), float(y2)]
        }