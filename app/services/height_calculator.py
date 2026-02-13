import numpy as np
import math


class HeightCalculator:
    def calculate(self, keypoints: dict, px_per_cm: float) -> dict:
        """
        키포인트와 비율(px_per_cm)을 받아 최종 키를 계산
        """
        if px_per_cm <= 0:
            return {"height_cm": 0, "error": "Invalid px_per_cm"}

        # 1. 머리 끝(Head Top) 추정 (COCO 모델은 정수리가 없음 -> 코에서 추정)
        # 코와 목(양 어깨 중점) 사이 거리만큼 코 위로 더해줌
        nose = keypoints["nose"]
        shoulder_center = (keypoints["left_shoulder"] + keypoints["right_shoulder"]) / 2

        neck_len_px = np.linalg.norm(nose - shoulder_center)
        # 영아는 머리가 크므로 1.5배 정도 보정
        head_top = nose - (shoulder_center - nose) * 0.8

        # 2. 세그먼트별 길이 합산 (다리 펴기)
        # (1) 머리끝 ~ 목
        seg1 = np.linalg.norm(head_top - shoulder_center)

        # (2) 목 ~ 엉덩이 중점 (몸통)
        hip_center = (keypoints["left_hip"] + keypoints["right_hip"]) / 2
        seg2 = np.linalg.norm(shoulder_center - hip_center)

        # (3) 다리 길이 (더 긴 쪽 선택 - 쭉 뻗은 다리가 정확함)
        left_leg = (np.linalg.norm(keypoints["left_hip"] - keypoints["left_knee"]) +
                    np.linalg.norm(keypoints["left_knee"] - keypoints["left_ankle"]))

        right_leg = (np.linalg.norm(keypoints["right_hip"] - keypoints["right_knee"]) +
                     np.linalg.norm(keypoints["right_knee"] - keypoints["right_ankle"]))

        seg3 = max(left_leg, right_leg)

        # (4) 발목 ~ 발바닥 보정 (약 3~4cm 추가)
        # YOLOv8-pose는 발끝 점이 없으므로 상수로 보정
        foot_correction_cm = 3.5
        foot_correction_px = foot_correction_cm * px_per_cm

        # 3. 최종 합산
        total_px = seg1 + seg2 + seg3 + foot_correction_px
        height_cm = total_px / px_per_cm

        return {
            "height_cm": round(height_cm, 1),
            "segments": {
                "head": round(seg1 / px_per_cm, 1),
                "torso": round(seg2 / px_per_cm, 1),
                "leg": round(seg3 / px_per_cm, 1)
            }
        }