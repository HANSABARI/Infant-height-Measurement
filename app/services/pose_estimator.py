from ultralytics import YOLO
import numpy as np
import cv2
import os


class PoseEstimator:
    def __init__(self, model_path: str = None):
        # 1. 모델 경로 설정 (없으면 yolov8n-pose.pt 자동 다운로드)
        if model_path is None:
            # 기본 모델: YOLOv8-Nano Pose
            self.model_name = "yolov8n-pose.pt"
        else:
            self.model_name = model_path

        print(f"Loading Pose model: {self.model_name}")
        self.model = YOLO(self.model_name)

    def estimate(self, img: np.ndarray) -> dict:
        """
        이미지에서 사람을 찾아 키포인트(관절) 좌표를 반환
        """
        # 추론 실행 (사람 1명만 찾도록 classes=0 설정 가능)
        results = self.model(img, verbose=False)

        # 사람이 안 보이면 실패 처리
        if not results or len(results[0].keypoints) == 0:
            return {"detected": False, "keypoints": {}, "confidence": 0.0}

        # 가장 크게 잡힌 사람 1명 선택
        # keypoints.data shape: (N, 17, 3) -> (x, y, conf)
        # COCO Keypoints: 0:nose, 5:L-shoulder, 6:R-shoulder, 11:L-hip, 12:R-hip, ...
        person_kpts = results[0].keypoints.data[0].cpu().numpy()

        # --- 키포인트 매핑 (나중에 RTMPose로 바꿔도 이 구조 유지) ---
        keypoints = {
            "nose": person_kpts[0][:2],
            "left_shoulder": person_kpts[5][:2],
            "right_shoulder": person_kpts[6][:2],
            "left_hip": person_kpts[11][:2],
            "right_hip": person_kpts[12][:2],
            "left_knee": person_kpts[13][:2],
            "right_knee": person_kpts[14][:2],
            "left_ankle": person_kpts[15][:2],
            "right_ankle": person_kpts[16][:2],
        }

        # 신뢰도 평균 계산
        confidences = person_kpts[:, 2]
        avg_conf = float(np.mean(confidences))

        return {
            "detected": True,
            "keypoints": keypoints,
            "confidence": avg_conf
        }


# --- 테스트 코드 (이 파일만 실행해보기 위해) ---
if __name__ == "__main__":
    estimator = PoseEstimator()
    # 검은 화면(0) 대신 임의의 이미지나 webcam으로 테스트 가능
    dummy_img = np.zeros((640, 480, 3), dtype=np.uint8)

    result = estimator.estimate(dummy_img)
    print("Pose Test Result:", result)