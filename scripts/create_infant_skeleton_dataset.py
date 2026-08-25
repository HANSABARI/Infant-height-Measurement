#!/usr/bin/env python3
"""
Create a skeleton-preannotated infant dataset with RTMPose WholeBody.

Outputs:
- copied source images
- preview images with skeleton overlays
- COCO-style keypoint annotations
- per-image JSON labels for lightweight inspection/editing
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# These names are exactly the keys added in app/services/pose_estimator.py
# keypoints_dict.update(...), with head_top appended for manual annotation.
POSE_ESTIMATOR_KEYPOINTS = [
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
    "head_top",
]

POSE_ESTIMATOR_INDEX_BY_NAME = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
    "left_eye": 1,
    "right_eye": 2,
    "left_heel": 19,
    "right_heel": 22,
}

POSE_CONFIDENCE_INDICES = [0, 5, 6, 11, 12, 13, 14, 15, 16]

SKELETON_CONNECTIONS = [
    ("left_eye", "nose"),
    ("right_eye", "nose"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
]


def iter_image_files(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_pose_estimator(device: str):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from app.services.pose_estimator import PoseEstimator
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import app.services.pose_estimator. Run this script from the "
            "h-align-server project root and install requirements.txt first."
        ) from exc

    return PoseEstimator(device=device)


def point_to_xy(point: Any) -> List[float]:
    if point is None:
        return [0.0, 0.0]
    if hasattr(point, "tolist"):
        point = point.tolist()
    if len(point) < 2:
        return [0.0, 0.0]
    x = float(point[0])
    y = float(point[1])
    return [x, y]


def build_named_pose_result(keypoints: Any, scores: Any) -> Dict[str, Any]:
    if len(keypoints) == 0:
        return {"detected": False, "keypoints": {}, "scores": {}, "confidence": 0.0}

    person_keypoints = keypoints[0]
    person_scores = scores[0]

    named_keypoints = {}
    named_scores = {}
    for name, index in POSE_ESTIMATOR_INDEX_BY_NAME.items():
        named_keypoints[name] = point_to_xy(person_keypoints[index])
        named_scores[name] = float(person_scores[index])

    confidence_scores = [float(person_scores[index]) for index in POSE_CONFIDENCE_INDICES]
    confidence = sum(confidence_scores) / len(confidence_scores)

    return {
        "detected": True,
        "keypoints": named_keypoints,
        "scores": named_scores,
        "confidence": confidence,
    }


def estimate_pose(estimator: Any, image: Any) -> Dict[str, Any]:
    if hasattr(estimator, "pose"):
        keypoints, scores = estimator.pose(image)
        return build_named_pose_result(keypoints, scores)
    return estimator.estimate(image)


def normalize_pose_result(
    pose_result: Dict[str, Any],
    keypoint_names: Sequence[str],
) -> Dict[str, Any]:
    source_keypoints = pose_result.get("keypoints") or {}
    source_scores = pose_result.get("scores") or {}
    default_score = float(pose_result.get("confidence") or 0.0)

    keypoints: List[List[float]] = []
    scores: List[float] = []

    for name in keypoint_names:
        point = source_keypoints.get(name)
        xy = point_to_xy(point)
        keypoints.append(xy)

        if point is None:
            scores.append(0.0)
        elif isinstance(source_scores, dict) and name in source_scores:
            scores.append(float(source_scores[name]))
        else:
            scores.append(default_score)

    return {
        "detected": bool(pose_result.get("detected")),
        "confidence": default_score,
        "keypoints": keypoints,
        "scores": scores,
    }


def visible_points(
    normalized: Dict[str, Any],
    score_threshold: float,
) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for (x, y), score in zip(normalized["keypoints"], normalized["scores"]):
        if score >= score_threshold and (x != 0.0 or y != 0.0):
            points.append((float(x), float(y)))
    return points


def bbox_from_points(points: Sequence[Tuple[float, float]]) -> List[float]:
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min = min(xs)
    y_min = min(ys)
    return [x_min, y_min, max(xs) - x_min, max(ys) - y_min]


def build_coco_annotation(
    annotation_id: int,
    image_id: int,
    normalized: Dict[str, Any],
    score_threshold: float,
) -> Dict[str, Any]:
    flat_keypoints: List[float] = []
    num_keypoints = 0

    for (x, y), score in zip(normalized["keypoints"], normalized["scores"]):
        if score >= score_threshold and (x != 0.0 or y != 0.0):
            flat_keypoints.extend([float(x), float(y), 2])
            num_keypoints += 1
        else:
            flat_keypoints.extend([0.0, 0.0, 0])

    bbox = bbox_from_points(visible_points(normalized, score_threshold))

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "iscrowd": 0,
        "segmentation": [],
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "num_keypoints": num_keypoints,
        "keypoints": flat_keypoints,
        "keypoint_scores": [float(score) for score in normalized["scores"]],
        "pose_confidence": float(normalized["confidence"]),
    }


def build_coco_skeleton(keypoint_names: Sequence[str]) -> List[List[int]]:
    name_to_index = {name: index + 1 for index, name in enumerate(keypoint_names)}
    skeleton = []
    for first, second in SKELETON_CONNECTIONS:
        if first in name_to_index and second in name_to_index:
            skeleton.append([name_to_index[first], name_to_index[second]])
    return skeleton


def draw_preview(
    image: Any,
    normalized: Dict[str, Any],
    keypoint_names: Sequence[str],
    score_threshold: float,
    draw_labels: bool,
) -> Any:
    import cv2

    preview = image.copy()
    name_to_point = {
        name: (point, score)
        for name, point, score in zip(
            keypoint_names, normalized["keypoints"], normalized["scores"]
        )
    }

    for first, second in SKELETON_CONNECTIONS:
        first_point = name_to_point.get(first)
        second_point = name_to_point.get(second)
        if first_point is None or second_point is None:
            continue
        (x1, y1), score1 = first_point
        (x2, y2), score2 = second_point
        if score1 < score_threshold or score2 < score_threshold:
            continue
        if (x1 == 0.0 and y1 == 0.0) or (x2 == 0.0 and y2 == 0.0):
            continue
        cv2.line(preview, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    for name, (point, score) in name_to_point.items():
        x, y = point
        if score < score_threshold or (x == 0.0 and y == 0.0):
            continue
        cv2.circle(preview, (int(x), int(y)), 5, (0, 0, 255), -1)
        if draw_labels:
            cv2.putText(
                preview,
                name,
                (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return preview


def write_label_json(
    path: Path,
    image_file_name: str,
    keypoint_names: Sequence[str],
    normalized: Dict[str, Any],
    score_threshold: float,
) -> None:
    labels = {}
    for name, point, score in zip(
        keypoint_names, normalized["keypoints"], normalized["scores"]
    ):
        x, y = point
        labels[name] = {
            "x": float(x),
            "y": float(y),
            "score": float(score),
            "visible": bool(score >= score_threshold and (x != 0.0 or y != 0.0)),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "image": image_file_name,
                "detected": bool(normalized["detected"]),
                "confidence": float(normalized["confidence"]),
                "keypoints": labels,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_dataset(
    input_dir: Path,
    output_dir: Path,
    device: str,
    score_threshold: float,
    limit: int | None,
    draw_labels: bool,
) -> Dict[str, Any]:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required. Install project dependencies first, for example: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_dir}")

    image_paths = iter_image_files(input_dir)
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")

    split_name = input_dir.name
    split_output_dir = output_dir / split_name
    images_output_dir = split_output_dir / "images"
    previews_output_dir = split_output_dir / "previews"
    labels_output_dir = split_output_dir / "labels"

    images_output_dir.mkdir(parents=True, exist_ok=True)
    previews_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    estimator = load_pose_estimator(device=device)

    coco_images = []
    coco_annotations = []
    failures = []

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(input_dir)
        image = cv2.imread(str(image_path))
        if image is None:
            failures.append({"file": relative_path.as_posix(), "reason": "cv2.imread failed"})
            continue

        pose_result = estimate_pose(estimator, image)
        normalized = normalize_pose_result(pose_result, POSE_ESTIMATOR_KEYPOINTS)

        copied_image_path = images_output_dir / relative_path
        copied_image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, copied_image_path)

        preview = draw_preview(
            image=image,
            normalized=normalized,
            keypoint_names=POSE_ESTIMATOR_KEYPOINTS,
            score_threshold=score_threshold,
            draw_labels=draw_labels,
        )
        preview_path = previews_output_dir / relative_path
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_path), preview)

        write_label_json(
            labels_output_dir / relative_path.with_suffix(".json"),
            copied_image_path.relative_to(split_output_dir).as_posix(),
            POSE_ESTIMATOR_KEYPOINTS,
            normalized,
            score_threshold,
        )

        height, width = image.shape[:2]
        coco_images.append(
            {
                "id": index,
                "file_name": copied_image_path.relative_to(split_output_dir).as_posix(),
                "width": int(width),
                "height": int(height),
            }
        )
        coco_annotations.append(
            build_coco_annotation(
                annotation_id=index,
                image_id=index,
                normalized=normalized,
                score_threshold=score_threshold,
            )
        )

        print(
            f"[{index}/{len(image_paths)}] {relative_path.as_posix()} "
            f"detected={normalized['detected']} confidence={normalized['confidence']:.3f}"
        )

    coco = {
        "info": {
            "description": "RTMPose skeleton preannotations for infant_dataset",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_input_dir": str(input_dir),
            "score_threshold": score_threshold,
            "note": "head_top is intentionally left unlabeled for manual annotation.",
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {
                "id": 1,
                "name": "infant",
                "supercategory": "person",
                "keypoints": POSE_ESTIMATOR_KEYPOINTS,
                "skeleton": build_coco_skeleton(POSE_ESTIMATOR_KEYPOINTS),
            }
        ],
    }

    annotations_path = split_output_dir / "annotations.json"
    annotations_path.write_text(
        json.dumps(coco, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(split_output_dir),
        "total_images": len(image_paths),
        "written_images": len(coco_images),
        "failures": failures,
        "keypoints": POSE_ESTIMATOR_KEYPOINTS,
        "annotations": str(annotations_path),
    }
    (split_output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return manifest


def cvat_image_relative_path(coco_file_name: str, subset_name: str) -> Path:
    image_path = Path(coco_file_name)
    parts = image_path.parts
    if parts and parts[0] == "images":
        image_path = Path(*parts[1:])
    return Path(subset_name) / image_path


def standard_coco_annotation(annotation: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "id",
        "image_id",
        "category_id",
        "segmentation",
        "area",
        "bbox",
        "iscrowd",
        "keypoints",
        "num_keypoints",
    ]
    return {key: copy.deepcopy(annotation[key]) for key in keys if key in annotation}


def build_cvat_coco_keypoints_archive(
    split_dir: Path,
    zip_path: Path,
    subset_name: str,
) -> Path:
    annotations_path = split_dir / "annotations.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing annotations file: {annotations_path}")

    coco = json.loads(annotations_path.read_text(encoding="utf-8"))
    cvat_coco = {
        "info": copy.deepcopy(coco.get("info", {})),
        "licenses": copy.deepcopy(coco.get("licenses", [])),
        "images": [],
        "annotations": [
            standard_coco_annotation(annotation)
            for annotation in coco.get("annotations", [])
        ],
        "categories": copy.deepcopy(coco.get("categories", [])),
    }

    image_entries = []
    for image in coco.get("images", []):
        cvat_relative_path = cvat_image_relative_path(image["file_name"], subset_name)
        image_entry = copy.deepcopy(image)
        image_entry["file_name"] = cvat_relative_path.as_posix()
        cvat_coco["images"].append(image_entry)

        source_path = split_dir / image["file_name"]
        if not source_path.exists():
            raise FileNotFoundError(f"Missing image referenced by annotations: {source_path}")
        archive_path = Path("images") / cvat_relative_path
        image_entries.append((source_path, archive_path.as_posix()))

    annotation_archive_path = f"annotations/person_keypoints_{subset_name}.json"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            annotation_archive_path,
            json.dumps(cvat_coco, ensure_ascii=False, indent=2),
        )
        for source_path, archive_path in image_entries:
            archive.write(source_path, archive_path)

    return zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create RTMPose skeleton preannotations for infant_dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset/infant_dataset/train"),
        help="Input image directory. Default: dataset/infant_dataset/train",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/infant_dataset_skel"),
        help="Output dataset directory. Default: dataset/infant_dataset_skel",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device passed to rtmlib Wholebody via PoseEstimator.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Minimum confidence for a keypoint to be marked visible.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N images. Useful for a quick smoke test.",
    )
    parser.add_argument(
        "--draw-labels",
        action="store_true",
        help="Draw keypoint names on preview images.",
    )
    parser.add_argument(
        "--cvat-zip",
        type=Path,
        default=None,
        help="Also write a CVAT COCO Keypoints zip archive at this path.",
    )
    parser.add_argument(
        "--package-cvat-only",
        action="store_true",
        help="Do not run RTMPose; package an existing output split into CVAT zip.",
    )
    parser.add_argument(
        "--subset-name",
        default=None,
        help="Subset name for CVAT zip. Defaults to the processed split directory name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.package_cvat_only:
        if args.cvat_zip is None:
            raise ValueError("--cvat-zip is required with --package-cvat-only")
        split_dir = args.output_dir / args.input_dir.name
        subset_name = args.subset_name or split_dir.name
        zip_path = build_cvat_coco_keypoints_archive(split_dir, args.cvat_zip, subset_name)
        print(f"Done. Wrote CVAT COCO Keypoints archive to {zip_path}")
        return

    manifest = build_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device=args.device,
        score_threshold=args.score_threshold,
        limit=args.limit,
        draw_labels=args.draw_labels,
    )
    print(f"Done. Wrote {manifest['written_images']} images to {manifest['output_dir']}")
    if manifest["failures"]:
        print(f"Skipped {len(manifest['failures'])} unreadable images. See manifest.json.")
    if args.cvat_zip is not None:
        split_dir = Path(manifest["output_dir"])
        subset_name = args.subset_name or split_dir.name
        zip_path = build_cvat_coco_keypoints_archive(split_dir, args.cvat_zip, subset_name)
        print(f"Wrote CVAT COCO Keypoints archive to {zip_path}")


if __name__ == "__main__":
    main()
