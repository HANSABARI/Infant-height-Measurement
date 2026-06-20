import os

import cv2
import numpy as np
import torch
from mmdet.apis import init_detector, inference_detector


_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


class CardDetector:
    def __init__(
        self,
        config_path: str = None,
        checkpoint_path: str = None,
        device: str = "cuda:0",
        score_threshold: float = 0.3,
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if config_path is None:
            config_path = os.getenv(
                "CARD_RTM_CONFIG",
                os.path.join(base_dir, "models", "card_model_v2", "rtmdet-ins_card.py"),
            )
        if checkpoint_path is None:
            checkpoint_path = os.getenv(
                "CARD_RTM_CHECKPOINT",
                os.path.join(base_dir, "models", "card_model_rtmdet-ins", "best_coco_segm_mAP_epoch_97.pth"),
                # os.path.join(base_dir, "models", "card_model_v2", "rtmdet_ins_epoch_300.pth"),
            )

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"RTMDet-Ins config not found: {config_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"RTMDet-Ins checkpoint not found: {checkpoint_path}")

        print(f"Loading RTMDet-Ins model...\nConfig: {config_path}\nWeights: {checkpoint_path}")
        self.model = init_detector(config_path, checkpoint_path, device=device)
        self.score_threshold = score_threshold

        self.CARD_WIDTH_CM = 8.56
        self.CARD_HEIGHT_CM = 5.4

    def detect(self, img: np.ndarray) -> dict:
        result = inference_detector(self.model, img)

        pred_instances = result.pred_instances
        scores = pred_instances.scores.cpu().numpy()
        bboxes = pred_instances.bboxes.cpu().numpy()

        valid_indices = scores > self.score_threshold
        if not valid_indices.any():
            return {
                "detected": False,
                "px_per_cm": 0.0,
                "confidence": 0.0,
                "bbox": [],
                "mask": None,
                "used_mask": False,
                "measurement_box": None,
                "long_side_segments": None,
            }

        valid_instance_indices = np.flatnonzero(valid_indices)
        best_idx = int(valid_instance_indices[scores[valid_indices].argmax()])

        x1, y1, x2, y2 = bboxes[best_idx]
        conf = float(scores[best_idx])

        w_px = x2 - x1
        h_px = y2 - y1

        mask = self._get_instance_mask(pred_instances, best_idx, img.shape[:2])
        mask_geometry = self._mask_geometry(mask)
        long_side_px = (
            mask_geometry["long_side_px"]
            if mask_geometry is not None
            else max(w_px, h_px)
        )
        px_per_cm = float(long_side_px / self.CARD_WIDTH_CM)

        return {
            "detected": True,
            "px_per_cm": px_per_cm,
            "confidence": conf,
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "mask": mask,
            "used_mask": mask_geometry is not None,
            "measurement_box": (
                mask_geometry["box"].tolist() if mask_geometry is not None else None
            ),
            "long_side_points": (
                mask_geometry["long_side_points"].tolist()
                if mask_geometry is not None
                else None
            ),
            "long_side_segments": (
                mask_geometry["long_side_segments"].tolist()
                if mask_geometry is not None
                else None
            ),
            "long_side_lengths_px": (
                mask_geometry["long_side_lengths_px"].tolist()
                if mask_geometry is not None
                else None
            ),
            "long_side_px": float(long_side_px),
            "reference_length_cm": self.CARD_WIDTH_CM,
            "geometry_method": (
                mask_geometry["method"] if mask_geometry is not None else "bbox"
            ),
        }

    def _get_instance_mask(
        self,
        pred_instances,
        instance_idx: int,
        image_shape: tuple[int, int],
    ) -> np.ndarray | None:
        masks = getattr(pred_instances, "masks", None)
        if masks is None:
            return None

        try:
            mask = masks[instance_idx]
        except (IndexError, TypeError):
            return None

        return self._mask_to_numpy(mask, image_shape)

    @staticmethod
    def _mask_geometry(mask: np.ndarray | None) -> dict | None:
        if mask is None:
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 10:
            return None

        rect = cv2.minAreaRect(contour)
        (_, _), (width, height), _ = rect
        if width <= 0 or height <= 0:
            return None

        initial_box = cv2.boxPoints(rect).astype(np.float32)
        contour_points = contour.reshape(-1, 2).astype(np.float32)
        fitted_lines = []

        for edge_idx in range(4):
            start = initial_box[edge_idx]
            end = initial_box[(edge_idx + 1) % 4]
            edge = end - start
            edge_length_sq = float(np.dot(edge, edge))
            if edge_length_sq <= 0:
                return CardDetector._min_area_geometry(initial_box)

            relative = contour_points - start
            projection = (relative @ edge) / edge_length_sq
            cross = np.abs(edge[0] * relative[:, 1] - edge[1] * relative[:, 0])
            distance = cross / np.sqrt(edge_length_sq)

            # Ignore rounded corners and fit only the central portion of each side.
            side_points = contour_points[
                (projection >= 0.12)
                & (projection <= 0.88)
                & (distance <= max(3.0, 0.08 * np.sqrt(edge_length_sq)))
            ]
            if len(side_points) < 6:
                return CardDetector._min_area_geometry(initial_box)

            fitted = cv2.fitLine(
                side_points.reshape(-1, 1, 2),
                cv2.DIST_HUBER,
                0,
                0.01,
                0.01,
            ).reshape(-1)
            fitted_lines.append(
                (
                    np.array([fitted[2], fitted[3]], dtype=np.float32),
                    np.array([fitted[0], fitted[1]], dtype=np.float32),
                )
            )

        corners = []
        for edge_idx in range(4):
            previous_line = fitted_lines[(edge_idx - 1) % 4]
            current_line = fitted_lines[edge_idx]
            corner = CardDetector._line_intersection(previous_line, current_line)
            if corner is None:
                return CardDetector._min_area_geometry(initial_box)
            corners.append(corner)

        box = np.asarray(corners, dtype=np.float32)
        contour_area = float(cv2.contourArea(contour))
        fitted_area = abs(float(cv2.contourArea(box)))
        if (
            not np.isfinite(box).all()
            or not cv2.isContourConvex(box.astype(np.int32))
            or fitted_area < contour_area * 0.65
            or fitted_area > contour_area * 1.35
        ):
            return CardDetector._min_area_geometry(initial_box)

        return CardDetector._geometry_from_box(box, method="robust_line_fit")

    @staticmethod
    def _line_intersection(
        first_line: tuple[np.ndarray, np.ndarray],
        second_line: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray | None:
        first_point, first_direction = first_line
        second_point, second_direction = second_line
        denominator = (
            first_direction[0] * second_direction[1]
            - first_direction[1] * second_direction[0]
        )
        if abs(float(denominator)) < 1e-6:
            return None

        delta = second_point - first_point
        distance = (
            delta[0] * second_direction[1]
            - delta[1] * second_direction[0]
        ) / denominator
        return first_point + distance * first_direction

    @staticmethod
    def _min_area_geometry(box: np.ndarray) -> dict:
        return CardDetector._geometry_from_box(
            box.astype(np.float32),
            method="min_area_rect_fallback",
        )

    @staticmethod
    def _geometry_from_box(box: np.ndarray, method: str) -> dict:
        edges = np.roll(box, -1, axis=0) - box
        edge_lengths = np.linalg.norm(edges, axis=1)
        first_pair_mean = float((edge_lengths[0] + edge_lengths[2]) / 2)
        second_pair_mean = float((edge_lengths[1] + edge_lengths[3]) / 2)
        long_edge_indices = (0, 2) if first_pair_mean >= second_pair_mean else (1, 3)

        long_side_segments = np.asarray(
            [
                [box[index], box[(index + 1) % 4]]
                for index in long_edge_indices
            ],
            dtype=np.float32,
        )
        long_side_lengths = edge_lengths[list(long_edge_indices)].astype(np.float32)
        long_side_px = float(np.mean(long_side_lengths))

        return {
            "box": box.astype(np.float32),
            "long_side_points": long_side_segments[0],
            "long_side_segments": long_side_segments,
            "long_side_lengths_px": long_side_lengths,
            "long_side_px": long_side_px,
            "method": method,
        }

    @staticmethod
    def _mask_to_numpy(mask, image_shape: tuple[int, int]) -> np.ndarray:
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        elif hasattr(mask, "to_ndarray"):
            mask = mask.to_ndarray()
        else:
            mask = np.asarray(mask)

        if mask.ndim == 3:
            mask = mask[0]

        mask = (mask > 0.5).astype(np.uint8)
        if mask.shape[:2] != image_shape:
            mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)

        return mask
