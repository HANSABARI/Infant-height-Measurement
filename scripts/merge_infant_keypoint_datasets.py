#!/usr/bin/env python3
"""Merge extracted infant COCO Keypoints dataset parts for RTMPose training."""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


CANONICAL_KEYPOINTS = [
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

SOURCE_DIR_PATTERN = re.compile(r"part[0-9]+_[0-9]+-[0-9]+$", re.IGNORECASE)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}

SKELETON = [
    [10, 1],
    [11, 1],
    [1, 2],
    [1, 3],
    [2, 3],
    [2, 4],
    [3, 5],
    [4, 5],
    [4, 6],
    [6, 8],
    [8, 12],
    [5, 7],
    [7, 9],
    [9, 13],
]


def discover_source_dirs(splits_dir: Path) -> list[Path]:
    """Return only compatible immediate child directories, sorted by name."""
    if not splits_dir.is_dir():
        raise FileNotFoundError(f"Splits directory does not exist: {splits_dir}")
    return sorted(
        path
        for path in splits_dir.iterdir()
        if path.is_dir() and SOURCE_DIR_PATTERN.search(path.name)
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON annotation file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"COCO annotation must be an object: {path}")
    return payload


def find_annotation_file(source_dir: Path) -> Path:
    """Find the source's COCO Keypoints JSON file."""
    annotation_dir = source_dir / "annotations"
    candidates = sorted(annotation_dir.glob("*.json"))
    candidates.sort(key=lambda path: (not path.name.startswith("person_keypoints"), path.name))
    valid = []
    for candidate in candidates:
        payload = _load_json(candidate)
        if all(key in payload for key in ("images", "annotations", "categories")):
            valid.append(candidate)
    if not valid:
        raise FileNotFoundError(f"No COCO Keypoints JSON found under: {annotation_dir}")
    if len(valid) > 1:
        preferred = [path for path in valid if path.name.startswith("person_keypoints")]
        if len(preferred) == 1:
            return preferred[0]
        raise ValueError(f"Multiple COCO annotation files found under: {annotation_dir}")
    return valid[0]


def resolve_image_path(source_dir: Path, file_name: str) -> Path:
    """Resolve an annotation image path across common CVAT export layouts."""
    relative = Path(file_name)
    parts_without_images = (
        Path(*relative.parts[1:]) if relative.parts and relative.parts[0] == "images" else relative
    )
    candidates = [
        source_dir / relative,
        source_dir / "images" / relative,
        source_dir / "images" / parts_without_images,
    ]
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate

    image_root = source_dir / "images"
    if image_root.is_dir():
        basename_matches = sorted(image_root.rglob(relative.name))
        if len(basename_matches) == 1:
            return basename_matches[0]
        if len(basename_matches) > 1:
            raise FileExistsError(
                f"Ambiguous image path '{file_name}' under {source_dir}: {basename_matches}"
            )
    raise FileNotFoundError(f"Image '{file_name}' not found under {source_dir}")


def _validate_keypoint_names(keypoint_names: list[str]) -> None:
    if len(keypoint_names) != len(CANONICAL_KEYPOINTS):
        raise ValueError(
            f"Expected {len(CANONICAL_KEYPOINTS)} keypoints, got {len(keypoint_names)}"
        )
    if set(keypoint_names) != set(CANONICAL_KEYPOINTS):
        missing = sorted(set(CANONICAL_KEYPOINTS) - set(keypoint_names))
        extra = sorted(set(keypoint_names) - set(CANONICAL_KEYPOINTS))
        raise ValueError(f"Incompatible keypoints; missing={missing}, extra={extra}")
    if len(set(keypoint_names)) != len(keypoint_names):
        raise ValueError("COCO category contains duplicate keypoint names")


def normalize_annotation(
    annotation: dict[str, Any],
    category: dict[str, Any],
    image_id: int,
    annotation_id: int,
) -> dict[str, Any]:
    """Copy one annotation while remapping its keypoints to canonical order."""
    source_names = list(category.get("keypoints") or [])
    _validate_keypoint_names(source_names)
    source_keypoints = list(annotation.get("keypoints") or [])
    expected_length = len(CANONICAL_KEYPOINTS) * 3
    if len(source_keypoints) != expected_length:
        raise ValueError(
            f"Annotation {annotation.get('id')} has {len(source_keypoints)} keypoint values; "
            f"expected {expected_length}"
        )

    source_index = {name: index for index, name in enumerate(source_names)}
    normalized_keypoints: list[Any] = []
    for name in CANONICAL_KEYPOINTS:
        start = source_index[name] * 3
        normalized_keypoints.extend(source_keypoints[start : start + 3])

    normalized = copy.deepcopy(annotation)
    normalized["id"] = annotation_id
    normalized["image_id"] = image_id
    normalized["category_id"] = 1
    normalized["keypoints"] = normalized_keypoints
    normalized["num_keypoints"] = sum(
        1
        for index in range(2, len(normalized_keypoints), 3)
        if float(normalized_keypoints[index]) > 0
    )
    return normalized


def _category_for_annotation(
    categories_by_id: dict[Any, dict[str, Any]], annotation: dict[str, Any]
) -> dict[str, Any]:
    category = categories_by_id.get(annotation.get("category_id"))
    if category is None:
        raise ValueError(
            f"Annotation {annotation.get('id')} references unknown category "
            f"{annotation.get('category_id')}"
        )
    return category


def _subset(coco: dict[str, Any], image_ids: set[int]) -> dict[str, Any]:
    result = copy.deepcopy(coco)
    result["images"] = [image for image in coco["images"] if image["id"] in image_ids]
    result["annotations"] = [
        annotation for annotation in coco["annotations"] if annotation["image_id"] in image_ids
    ]
    return result


def _output_image_name(source_dir: Path, source_image: Path) -> str:
    images_root = source_dir / "images"
    try:
        relative = source_image.relative_to(images_root)
        relative_name = "__".join(relative.parts)
    except ValueError:
        relative_name = source_image.name
    return f"{source_dir.name}__{relative_name}"


def merge_datasets(
    splits_dir: Path,
    output_root: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Merge all compatible extracted directories under ``splits_dir``."""
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be greater than or equal to 0 and less than 1")

    source_dirs = discover_source_dirs(splits_dir)
    if not source_dirs:
        raise ValueError(f"No compatible dataset directories found under: {splits_dir}")

    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    try:
        image_output = staging_root / "images"
        annotation_output = staging_root / "annotations"
        image_output.mkdir(parents=True)
        annotation_output.mkdir()

        merged = {
            "info": {"description": "Merged infant COCO Keypoints dataset"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "infant",
                    "supercategory": "person",
                    "keypoints": CANONICAL_KEYPOINTS,
                    "skeleton": SKELETON,
                }
            ],
        }
        next_image_id = 1
        next_annotation_id = 1
        source_summaries = []
        missing_keypoint_counts: dict[str, dict[str, int]] = {}

        for source_dir in source_dirs:
            annotation_path = find_annotation_file(source_dir)
            payload = _load_json(annotation_path)
            categories_by_id = {category["id"]: category for category in payload["categories"]}
            for category in categories_by_id.values():
                _validate_keypoint_names(list(category.get("keypoints") or []))

            image_id_map: dict[Any, int] = {}
            source_missing = {name: 0 for name in CANONICAL_KEYPOINTS}
            source_annotation_count = 0
            for image in payload["images"]:
                if "id" not in image or "file_name" not in image:
                    raise ValueError(f"Invalid image entry in {annotation_path}: {image}")
                source_image = resolve_image_path(source_dir, image["file_name"])
                output_name = _output_image_name(source_dir, source_image)
                shutil.copy2(source_image, image_output / output_name)

                new_image = copy.deepcopy(image)
                old_image_id = image["id"]
                image_id_map[old_image_id] = next_image_id
                new_image["id"] = next_image_id
                # `data_prefix.img` already points at the merged `images/`
                # directory, so COCO file names must be relative to it.
                new_image["file_name"] = output_name
                new_image["source_dataset"] = source_dir.name
                new_image["source_file_name"] = image["file_name"]
                merged["images"].append(new_image)
                next_image_id += 1

            for annotation in payload["annotations"]:
                old_image_id = annotation.get("image_id")
                if old_image_id not in image_id_map:
                    raise ValueError(
                        f"Annotation {annotation.get('id')} references unknown image "
                        f"{old_image_id} in {annotation_path}"
                    )
                category = _category_for_annotation(categories_by_id, annotation)
                normalized = normalize_annotation(
                    annotation,
                    category,
                    image_id=image_id_map[old_image_id],
                    annotation_id=next_annotation_id,
                )
                for index, name in enumerate(CANONICAL_KEYPOINTS):
                    if float(normalized["keypoints"][index * 3 + 2]) <= 0:
                        source_missing[name] += 1
                merged["annotations"].append(normalized)
                next_annotation_id += 1
                source_annotation_count += 1

            nonzero_missing = {
                name: count for name, count in source_missing.items() if count > 0
            }
            if nonzero_missing:
                missing_keypoint_counts[source_dir.name] = nonzero_missing
            source_summaries.append(
                {
                    "name": source_dir.name,
                    "annotation_file": str(annotation_path),
                    "images": len(payload["images"]),
                    "annotations": source_annotation_count,
                }
            )

        rng = random.Random(seed)
        image_ids = [image["id"] for image in merged["images"]]
        rng.shuffle(image_ids)
        val_count = 0 if val_ratio == 0 or not image_ids else max(1, round(len(image_ids) * val_ratio))
        val_ids = set(image_ids[:val_count])
        train_ids = set(image_ids[val_count:])

        merged["info"].update({"seed": seed, "val_ratio": val_ratio})
        outputs = {
            "person_keypoints.json": merged,
            "person_keypoints_train.json": _subset(merged, train_ids),
            "person_keypoints_val.json": _subset(merged, val_ids),
        }
        for filename, payload in outputs.items():
            (annotation_output / filename).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        manifest = {
            "splits_dir": str(splits_dir.resolve()),
            "output_root": str(output_root),
            "source_count": len(source_dirs),
            "sources": source_summaries,
            "keypoints": CANONICAL_KEYPOINTS,
            "missing_keypoint_counts": missing_keypoint_counts,
            "total_images": len(merged["images"]),
            "total_annotations": len(merged["annotations"]),
            "train_images": len(train_ids),
            "val_images": len(val_ids),
            "seed": seed,
            "val_ratio": val_ratio,
        }
        (staging_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if output_root.exists():
            shutil.rmtree(output_root)
        staging_root.rename(output_root)
        staging_root = None
    finally:
        if staging_root is not None and staging_root.exists():
            shutil.rmtree(staging_root)

    print(f"output: {output_root}")
    for summary in source_summaries:
        print(
            f"{summary['name']}: images={summary['images']} "
            f"annotations={summary['annotations']}"
        )
    print(
        f"merged: images={manifest['total_images']} "
        f"annotations={manifest['total_annotations']}"
    )
    print(f"train: images={manifest['train_images']}")
    print(f"val: images={manifest['val_images']}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge extracted infant COCO Keypoints dataset directories."
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("dataset/infant_dataset_skel/splits"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/infant_dataset_skel/merged"),
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_datasets(args.splits_dir, args.output_root, args.val_ratio, args.seed)
