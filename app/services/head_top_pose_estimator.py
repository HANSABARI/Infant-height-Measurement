import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from app.services.pose_estimator import PoseEstimator


WHOLEBODY_MEASUREMENT_KEYPOINT_NAMES = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_eye",
    "right_eye",
    "left_heel",
    "right_heel",
)
HEAD_TOP_KEYPOINT_NAME = "head_top"


class HeadTopPoseEstimator:
    """Keep WholeBody joints and add a separately trained head_top prediction."""

    def __init__(
        self,
        device: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        person_proposer: Optional[PoseEstimator] = None,
        model: Optional[Any] = None,
        model_loader: Optional[Callable[..., Any]] = None,
        inference_fn: Optional[Callable[..., Any]] = None,
        min_head_top_score: Optional[float] = None,
    ):
        project_root = Path(__file__).resolve().parents[2]
        default_work_dir = project_root / "app" / "models" / "rtmpose_infant_head_top"
        default_config = project_root / "app" / "configs" / "rtmpose_head_top.py"
        default_checkpoint = self.find_best_checkpoint(default_work_dir) or (
            default_work_dir / "best_coco_AP_epoch_100.pth"
        )

        self.device = device or os.getenv("INFANT_HEAD_TOP_DEVICE") or self._default_device()
        self.config_path = Path(
            config_path
            or os.getenv("INFANT_HEAD_TOP_CONFIG")
            or default_config
        )
        checkpoint_override = checkpoint_path or os.getenv("INFANT_HEAD_TOP_CHECKPOINT")
        self.uses_default_checkpoint = checkpoint_override is None
        self.default_work_dir = default_work_dir
        self.checkpoint_path = Path(checkpoint_override or default_checkpoint)
        self.person_proposer = person_proposer or PoseEstimator(device=self.device)
        self.min_head_top_score = (
            min_head_top_score
            if min_head_top_score is not None
            else float(os.getenv("INFANT_HEAD_TOP_MIN_SCORE", "0.2"))
        )

        if inference_fn is None:
            from mmpose.apis import inference_topdown

            inference_fn = inference_topdown
        self.inference_fn = inference_fn

        self.model = model
        self.model_loader = model_loader

    @staticmethod
    def find_best_checkpoint(work_dir: Path) -> Optional[Path]:
        candidates = list(work_dir.glob("best_coco_AP_epoch_*.pth"))
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _load_model(self) -> None:
        if self.model is not None:
            return
        if self.uses_default_checkpoint:
            self.checkpoint_path = self.find_best_checkpoint(self.default_work_dir) or (
                self.default_work_dir / "best_coco_AP_epoch_100.pth"
            )
        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"head_top RTMPose runtime config was not found: {self.config_path}"
            )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"head_top RTMPose checkpoint was not found: {self.checkpoint_path}"
            )
        if self.model_loader is None:
            from mmpose.apis import init_model

            self.model_loader = init_model
        print(
            "Loading head_top-only RTMPose checkpoint "
            f"on {self.device.upper()}: {self.checkpoint_path.name}"
        )
        self.model = self.model_loader(
            str(self.config_path),
            str(self.checkpoint_path),
            device=self.device,
        )

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except (AttributeError, ImportError):
            pass
        return "cpu"

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    @staticmethod
    def _padded_bbox(bbox: Any, image: np.ndarray) -> list[float]:
        bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if bbox.size != 4 or not np.isfinite(bbox).all():
            return []
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        if width <= 0 or height <= 0:
            return []

        image_height, image_width = image.shape[:2]
        return [
            float(max(0.0, x_min - width * 0.10)),
            float(max(0.0, y_min - height * 0.10)),
            float(min(float(image_width - 1), x_max + width * 0.10)),
            float(min(float(image_height - 1), y_max + height * 0.10)),
        ]

    @staticmethod
    def _not_detected(reason: str) -> Dict[str, Any]:
        return {
            "detected": False,
            "keypoints": {},
            "keypoint_scores": {},
            "confidence": 0.0,
            "bbox": [],
            "model": "RTMPose WholeBody + Head Top",
            "reason": reason,
        }

    def estimate(self, image: np.ndarray) -> Dict[str, Any]:
        wholebody_result = self.person_proposer.estimate(image)
        if not wholebody_result.get("detected"):
            return self._not_detected("wholebody_person_not_found")

        wholebody_keypoints = wholebody_result.get("keypoints", {})
        missing = [
            name
            for name in WHOLEBODY_MEASUREMENT_KEYPOINT_NAMES
            if name not in wholebody_keypoints
        ]
        if missing:
            return self._not_detected("wholebody_keypoints_missing")

        bbox = self._padded_bbox(wholebody_result.get("bbox"), image)
        if not bbox:
            return self._not_detected("wholebody_bbox_invalid")

        try:
            self._load_model()
        except FileNotFoundError:
            return self._not_detected("head_top_checkpoint_missing")

        results = self.inference_fn(
            self.model,
            image,
            bboxes=np.asarray([bbox], dtype=np.float32),
        )
        if not results:
            return self._not_detected("head_top_not_found")

        pred_instances = results[0].pred_instances
        points = self._to_numpy(pred_instances.keypoints)
        scores = self._to_numpy(pred_instances.keypoint_scores)
        if points.ndim == 3:
            points = points[0]
        if scores.ndim == 2:
            scores = scores[0]
        if points.shape != (1, 2) or scores.shape != (1,):
            return self._not_detected("unexpected_head_top_shape")

        head_top_score = float(np.clip(scores[0], 0.0, 1.0))
        if head_top_score < self.min_head_top_score:
            return self._not_detected("head_top_low_confidence")

        keypoints = {
            name: np.asarray(wholebody_keypoints[name], dtype=np.float32)
            for name in WHOLEBODY_MEASUREMENT_KEYPOINT_NAMES
        }
        keypoints[HEAD_TOP_KEYPOINT_NAME] = points[0].astype(np.float32)
        keypoint_scores = dict(wholebody_result.get("keypoint_scores", {}))
        keypoint_scores[HEAD_TOP_KEYPOINT_NAME] = head_top_score
        wholebody_confidence = float(
            np.clip(wholebody_result.get("confidence", 0.0), 0.0, 1.0)
        )

        return {
            "detected": True,
            "keypoints": keypoints,
            "keypoint_scores": keypoint_scores,
            "confidence": min(wholebody_confidence, head_top_score),
            "bbox": bbox,
            "model": "RTMPose WholeBody + Head Top",
            "checkpoint": str(self.checkpoint_path),
            "wholebody_confidence": wholebody_confidence,
        }
