from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np

from app.services.card_geometry import compute_mask_geometry, mask_to_numpy


CARD_REFERENCE_WIDTH_CM = 8.56
CARD_CLASS_IDS = {
    "id_card": 3,
    "passport": 5,
    "employee_badge": 6,
    "bank_card": 8,
}
MEASUREMENT_CARD_CLASS_NAMES = ("id_card", "bank_card")
DEFAULT_HAS_REPO_ID = "xuanwulab/HaS_Image_0209_FP32"
DEFAULT_HAS_MODEL_NAME = "HaS Image Model (FP32)"
DEFAULT_HAS_MODEL_FILENAME = "sensitive_seg_best.pt"
ULTRALYTICS_CONFIG_DIR_ENV = "YOLO_CONFIG_DIR"


class HaSCardDetector:
    def __init__(
        self,
        model_path: str | None = None,
        model=None,
        device: str | None = None,
        imgsz: int | None = None,
        tile_regions: list[tuple[int, int, int, int]] | None = None,
        tile_imgsz: int | None = None,
        score_threshold: float = 0.25,
        geometry_fn: Callable[[np.ndarray], dict | None] = compute_mask_geometry,
    ):
        base_dir = Path(__file__).resolve().parents[1]
        default_model_path = (
            base_dir
            / "models"
            / "has_image_0209_fp32"
            / DEFAULT_HAS_MODEL_FILENAME
        )

        self.model_path = model_path or os.getenv("CARD_HAS_MODEL", str(default_model_path))
        self.device = device if device is not None else os.getenv("CARD_HAS_DEVICE")
        self.imgsz = imgsz if imgsz is not None else int(os.getenv("CARD_HAS_IMGSZ", "1920"))
        self.tile_imgsz = (
            tile_imgsz
            if tile_imgsz is not None
            else int(os.getenv("CARD_HAS_TILE_IMGSZ", "1600"))
        )
        self.tile_regions = tile_regions
        self.score_threshold = score_threshold
        self.geometry_fn = geometry_fn
        self.card_class_ids = {
            CARD_CLASS_IDS[class_name]
            for class_name in MEASUREMENT_CARD_CLASS_NAMES
        }
        self.model = model if model is not None else self._load_model()

    def detect(self, img: np.ndarray) -> dict:
        result = self._predict(img, self.imgsz)
        if result is None:
            return self._empty_result()

        detected = self._detect_from_result(
            result=result,
            source_shape=img.shape[:2],
            original_shape=img.shape[:2],
            offset=(0, 0),
            detection_source="full",
            tile_region=None,
        )
        if detected is not None:
            return detected

        for tile_region in self._tile_regions(img.shape[:2]):
            x1, y1, x2, y2 = tile_region
            tile = img[y1:y2, x1:x2]
            if tile.size == 0:
                continue

            result = self._predict(tile, self.tile_imgsz)
            if result is None:
                continue

            detected = self._detect_from_result(
                result=result,
                source_shape=tile.shape[:2],
                original_shape=img.shape[:2],
                offset=(x1, y1),
                detection_source="tile",
                tile_region=tile_region,
            )
            if detected is not None:
                return detected

        return self._empty_result()

    def _predict(self, source: np.ndarray, imgsz: int):
        predict_kwargs = {
            "source": source,
            "conf": self.score_threshold,
            "imgsz": imgsz,
            "verbose": False,
        }
        if self.device:
            predict_kwargs["device"] = self.device

        results = self.model.predict(**predict_kwargs)
        return results[0] if results else None

    def _detect_from_result(
        self,
        result,
        source_shape: tuple[int, int],
        original_shape: tuple[int, int],
        offset: tuple[int, int],
        detection_source: str,
        tile_region: tuple[int, int, int, int] | None,
    ) -> dict | None:
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None or masks is None:
            return None

        classes = _to_numpy(getattr(boxes, "cls", [])).astype(int).reshape(-1)
        confidences = _to_numpy(getattr(boxes, "conf", [])).astype(float).reshape(-1)
        bboxes = _to_numpy(getattr(boxes, "xyxy", [])).astype(float)
        mask_data = _to_numpy(getattr(masks, "data", []))

        if (
            len(classes) == 0
            or len(confidences) != len(classes)
            or len(bboxes) != len(classes)
            or len(mask_data) != len(classes)
        ):
            return None

        valid_indices = [
            index
            for index, class_id in enumerate(classes)
            if class_id in self.card_class_ids
            and confidences[index] >= self.score_threshold
        ]
        if not valid_indices:
            return None

        best_idx = max(valid_indices, key=lambda index: confidences[index])
        source_mask = mask_to_numpy(mask_data[best_idx], source_shape)
        mask = self._place_mask(source_mask, original_shape, offset)
        mask_geometry = self.geometry_fn(mask)
        if mask_geometry is None:
            return None

        long_side_px = float(mask_geometry["long_side_px"])
        px_per_cm = float(long_side_px / CARD_REFERENCE_WIDTH_CM)
        offset_x, offset_y = offset
        x1, y1, x2, y2 = bboxes[best_idx]
        x1 += offset_x
        x2 += offset_x
        y1 += offset_y
        y2 += offset_y
        class_id = int(classes[best_idx])

        detected = {
            "detected": True,
            "px_per_cm": px_per_cm,
            "confidence": float(confidences[best_idx]),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "mask": mask,
            "used_mask": True,
            "measurement_box": mask_geometry["box"].tolist(),
            "long_side_points": mask_geometry["long_side_points"].tolist(),
            "long_side_segments": mask_geometry["long_side_segments"].tolist(),
            "long_side_lengths_px": mask_geometry["long_side_lengths_px"].tolist(),
            "long_side_px": long_side_px,
            "reference_length_cm": CARD_REFERENCE_WIDTH_CM,
            "geometry_method": mask_geometry["method"],
            "class_id": class_id,
            "class_name": self._class_name(result, class_id),
            "model": DEFAULT_HAS_MODEL_NAME,
            "model_repo": DEFAULT_HAS_REPO_ID,
            "detection_source": detection_source,
            "tile_region": list(tile_region) if tile_region is not None else None,
        }
        return detected

    @staticmethod
    def _place_mask(
        mask: np.ndarray,
        original_shape: tuple[int, int],
        offset: tuple[int, int],
    ) -> np.ndarray:
        offset_x, offset_y = offset
        if offset_x == 0 and offset_y == 0 and mask.shape[:2] == original_shape:
            return mask

        full_mask = np.zeros(original_shape, dtype=np.uint8)
        height, width = mask.shape[:2]
        full_mask[offset_y:offset_y + height, offset_x:offset_x + width] = mask
        return full_mask

    def _tile_regions(self, image_shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
        if self.tile_regions is not None:
            return self.tile_regions

        height, width = image_shape
        regions = [
            (width // 3, height // 2, (2 * width) // 3, height),
            (0, height // 2, width // 2, height),
            (width // 2, height // 2, width, height),
            (width // 4, height // 3, (3 * width) // 4, (5 * height) // 6),
            (0, height // 3, width // 2, (5 * height) // 6),
            (width // 2, height // 3, width, (5 * height) // 6),
        ]
        return [
            (x1, y1, x2, y2)
            for x1, y1, x2, y2 in regions
            if x2 > x1 and y2 > y1
        ]

    def _load_model(self):
        model_path = self._resolve_model_path()
        _ensure_ultralytics_config_dir()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for HaS Image Model (FP32) card segmentation. "
                "Install project requirements before starting the server."
            ) from exc

        print(
            f"Loading {DEFAULT_HAS_MODEL_NAME} segmentation model...\n"
            f"Weights: {model_path}\n"
            f"imgsz: {self.imgsz}, score_threshold: {self.score_threshold}"
        )
        return YOLO(model_path)

    def _resolve_model_path(self) -> str:
        model_path = Path(self.model_path).expanduser()
        if model_path.exists():
            return str(model_path)

        if os.getenv("CARD_HAS_AUTO_DOWNLOAD", "1") == "0":
            raise FileNotFoundError(self._missing_model_message(model_path))

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise FileNotFoundError(self._missing_model_message(model_path)) from exc

        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return hf_hub_download(
                repo_id=DEFAULT_HAS_REPO_ID,
                filename=DEFAULT_HAS_MODEL_FILENAME,
                local_dir=str(model_path.parent),
            )
        except Exception as exc:
            raise FileNotFoundError(self._missing_model_message(model_path)) from exc

    @staticmethod
    def _missing_model_message(model_path: Path) -> str:
        return (
            f"{DEFAULT_HAS_MODEL_NAME} segmentation weights not found: {model_path}. "
            "Download sensitive_seg_best.pt from "
            "https://huggingface.co/xuanwulab/HaS_Image_0209_FP32 "
            "or set CARD_HAS_MODEL to a local .pt path."
        )

    def _class_name(self, result, class_id: int) -> str:
        names = getattr(result, "names", None) or getattr(self.model, "names", None) or {}
        return names.get(class_id, str(class_id))

    @staticmethod
    def _empty_result() -> dict:
        return {
            "detected": False,
            "px_per_cm": 0.0,
            "confidence": 0.0,
            "bbox": [],
            "mask": None,
            "used_mask": False,
            "measurement_box": None,
            "long_side_points": None,
            "long_side_segments": None,
            "long_side_lengths_px": None,
            "long_side_px": 0.0,
            "reference_length_cm": CARD_REFERENCE_WIDTH_CM,
            "geometry_method": None,
            "class_id": None,
            "class_name": None,
            "model": DEFAULT_HAS_MODEL_NAME,
            "model_repo": DEFAULT_HAS_REPO_ID,
            "detection_source": None,
            "tile_region": None,
        }


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _ensure_ultralytics_config_dir() -> None:
    if os.getenv(ULTRALYTICS_CONFIG_DIR_ENV):
        return

    config_dir = Path(__file__).resolve().parents[1] / ".runtime" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ[ULTRALYTICS_CONFIG_DIR_ENV] = str(config_dir)
