from rtmlib import Wholebody
import numpy as np
import cv2
import os


class PoseEstimator:
    PROPOSAL_KEYPOINT_SCORE_THRESHOLD = 0.2
    def __init__(self, device: str = 'cpu'):
        """
        rtmlib를 활용하여 RTMPose-WholeBody ONNX 모델을 로드합니다.
        처음 실행 시 자동으로 최적화된 ONNX 모델을 다운로드합니다.
        """
        self.device = device
        print(f"Loading RTMPose-WholeBody model via rtmlib on {self.device.upper()}...")

        # mode: 'balanced'(기본), 'performance'(정확도 위주), 'lightweight'(속도 위주)
        self.pose = Wholebody(
            mode='balanced',
            backend='onnxruntime',
            device=self.device
        )

    def estimate(self, img: np.ndarray) -> dict:
        """
        이미지에서 사람을 찾아 키포인트(관절) 좌표를 반환합니다.
        """
        # 추론 실행 (rtmlib 내부에 사람 검출 -> 포즈 추정 파이프라인이 포함되어 있음)
        # keypoints shape: (N, 133, 2) -> (x, y)
        # scores shape: (N, 133) -> 각 키포인트의 신뢰도
        keypoints, scores = self.pose(img)

        # 사람이 안 보이면 실패 처리
        if len(keypoints) == 0:
            return {"detected": False, "keypoints": {}, "confidence": 0.0}

        # 가장 첫 번째 사람 선택
        # (rtmlib는 일반적으로 크기나 신뢰도 기준으로 N명을 반환하므로 0번 인덱스 사용)
        person_kpts = keypoints[0]
        person_scores = scores[0]

        # --- 키포인트 매핑 ---
        # COCO-WholeBody 133의 앞 17개 인덱스는 기존 COCO 17(YOLO)과 완벽히 동일합니다.
        # 따라서 기존 인덱스 번호를 그대로 유지할 수 있습니다.
        # --- 키포인트 매핑 ---

        # 1. 133개 전체를 pt_0 ~ pt_132 이름으로 다 넣기
        keypoints_dict = {f"pt_{i}": person_kpts[i] for i in range(133)}

        # 2. 기존 디버그(선 그리기) 코드가 에러 나지 않도록 필수 9개는 원래 이름으로도 덮어쓰기
        keypoints_dict.update({
            "nose": person_kpts[0],
            "left_shoulder": person_kpts[5],
            "right_shoulder": person_kpts[6],
            "left_hip": person_kpts[11],
            "right_hip": person_kpts[12],
            "left_knee": person_kpts[13],
            "right_knee": person_kpts[14],
            "left_ankle": person_kpts[15],
            "right_ankle": person_kpts[16],
            "left_eye": person_kpts[1],
            "right_eye": person_kpts[2],
            "left_heel": person_kpts[19],
            "right_heel": person_kpts[22],
        })

        # 전체 키포인트가 아닌, 키 측정에 사용된 주요 관절들의 신뢰도 평균만 계산
        target_indices = [0, 5, 6, 11, 12, 13, 14, 15, 16]
        target_scores = person_scores[target_indices]
        avg_conf = float(np.mean(target_scores))
        bbox = self._bbox_from_scored_keypoints(person_kpts, person_scores)

        return {
            "detected": True,
            "keypoints": keypoints_dict,
            "confidence": avg_conf,
            "bbox": bbox,
        }

    @classmethod
    def _bbox_from_scored_keypoints(
        cls,
        keypoints: np.ndarray,
        scores: np.ndarray,
    ) -> list[float]:
        keypoints = np.asarray(keypoints, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        valid = (
            np.isfinite(keypoints).all(axis=1)
            & np.isfinite(scores)
            & (scores >= cls.PROPOSAL_KEYPOINT_SCORE_THRESHOLD)
        )
        confident_points = keypoints[valid]
        if len(confident_points) < 2:
            return []

        x_min, y_min = confident_points.min(axis=0)
        x_max, y_max = confident_points.max(axis=0)
        if x_max <= x_min or y_max <= y_min:
            return []
        return [float(x_min), float(y_min), float(x_max), float(y_max)]


# --- 테스트 코드 ---
if __name__ == "__main__":
    # GPU 환경(RunPod 등)에서는 device='cuda'로 변경
    estimator = PoseEstimator(device='cpu')

    # 검은 화면(0) 대신 임의의 이미지 생성 (테스트용)
    dummy_img = np.zeros((640, 480, 3), dtype=np.uint8)

    result = estimator.estimate(dummy_img)
    print("Pose Test Result:", result)
