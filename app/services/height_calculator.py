import numpy as np


class HeightCalculator:
    REQUIRED_TORSO_POINTS = (
        "head_top",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    )

    @staticmethod
    def _point(keypoints: dict, name: str):
        point = keypoints.get(name)
        if point is None:
            return None
        point = np.asarray(point, dtype=np.float32).reshape(-1)
        if point.size < 2 or not np.isfinite(point[:2]).all():
            return None
        if np.allclose(point[:2], 0.0):
            return None
        return point[:2]

    def _leg_length(self, keypoints: dict, side: str):
        names = (f"{side}_hip", f"{side}_knee", f"{side}_ankle", f"{side}_heel")
        points = [self._point(keypoints, name) for name in names]
        if any(point is None for point in points):
            return None
        return sum(np.linalg.norm(points[index] - points[index + 1]) for index in range(3))

    def calculate(self, keypoints: dict, px_per_cm: float) -> dict:
        """
        실제 head_top 및 heel 키포인트와 카드 비율(px_per_cm)로 신장을 계산합니다.
        """
        if px_per_cm <= 0:
            return {"height_cm": 0, "error": "Invalid px_per_cm"}

        points = {
            name: self._point(keypoints, name) for name in self.REQUIRED_TORSO_POINTS
        }
        missing_torso = [name for name, point in points.items() if point is None]
        if missing_torso:
            return {
                "height_cm": 0,
                "error": f"Missing required keypoints: {', '.join(missing_torso)}",
            }

        shoulder_center = (points["left_shoulder"] + points["right_shoulder"]) / 2
        hip_center = (points["left_hip"] + points["right_hip"]) / 2
        head_length_px = np.linalg.norm(points["head_top"] - shoulder_center)
        torso_length_px = np.linalg.norm(shoulder_center - hip_center)

        leg_lengths = [
            leg_length
            for leg_length in (
                self._leg_length(keypoints, "left"),
                self._leg_length(keypoints, "right"),
            )
            if leg_length is not None
        ]
        if not leg_lengths:
            return {
                "height_cm": 0,
                "error": "Missing a complete hip-knee-ankle-heel leg chain",
            }

        leg_length_px = max(leg_lengths)
        total_px = head_length_px + torso_length_px + leg_length_px
        height_cm = total_px / px_per_cm

        return {
            "height_cm": round(height_cm, 1),
            "segments": {
                "head": round(head_length_px / px_per_cm, 1),
                "torso": round(torso_length_px / px_per_cm, 1),
                "leg": round(leg_length_px / px_per_cm, 1),
                "foot": 0.0,
            },
        }
