import cv2
import numpy as np


class Visualizer:
    def draw_debug(self, img: np.ndarray, card_result: dict, pose_result: dict) -> np.ndarray:
        """
        원본 이미지 위에 카드 박스와 관절 스켈레톤을 그려서 반환
        """
        debug_img = img.copy()

        # 1. 카드 그리기 (초록색 박스)
        if card_result["detected"]:
            x1, y1, x2, y2 = map(int, card_result["bbox"])
            # 박스 그리기 (BGR: 초록색)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 텍스트
            label = f"Card ({card_result['confidence']:.2f})"
            cv2.putText(debug_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 2. 관절 그리기 (빨간색 점 & 선)
        if pose_result["detected"]:
            kpts = pose_result["keypoints"]

            # (1) 점 찍기
            for name, point in kpts.items():
                x, y = map(int, point)
                if x == 0 and y == 0: continue  # 감지 안 된 점 패스
                cv2.circle(debug_img, (x, y), 5, (0, 0, 255), -1)  # 빨간 점

            # (2) 뼈대 잇기 (Skeleton Lines)
            # 연결할 부위 쌍 정의
            connections = [
                ("left_shoulder", "right_shoulder"),
                ("left_shoulder", "left_hip"),
                ("right_shoulder", "right_hip"),
                ("left_hip", "right_hip"),  # 골반
                ("left_hip", "left_knee"),  # 왼 다리 위
                ("left_knee", "left_ankle"),  # 왼 다리 아래
                ("right_hip", "right_knee"),  # 오른 다리 위
                ("right_knee", "right_ankle"),  # 오른 다리 아래
                ("nose", "left_shoulder"),  # 목 (가상 연결)
                ("nose", "right_shoulder")
            ]

            for p1_name, p2_name in connections:
                if p1_name in kpts and p2_name in kpts:
                    pt1 = tuple(map(int, kpts[p1_name]))
                    pt2 = tuple(map(int, kpts[p2_name]))
                    # 0,0 좌표가 아닐 때만 선 긋기
                    if pt1 != (0, 0) and pt2 != (0, 0):
                        cv2.line(debug_img, pt1, pt2, (255, 0, 0), 2)  # 파란 선

        return debug_img