#!/usr/bin/env python3
"""Train a single-keypoint RTMPose model for infant head_top only."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_rtmpose_infant import (  # noqa: E402
    DEFAULT_WARMUP_EPOCHS,
    build_runtime_config,
    find_default_base_config,
    run_training,
    validate_dataset_root,
    write_runtime_config,
)


HEAD_TOP_KEYPOINT_NAME = "head_top"
TRAIN_ANNOTATION_NAME = "person_keypoints_head_top_train.json"
VAL_ANNOTATION_NAME = "person_keypoints_head_top_val.json"


def _load_coco(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid COCO annotation JSON: {path}") from exc
    if not all(key in payload for key in ("images", "annotations", "categories")):
        raise ValueError(f"Incomplete COCO annotation JSON: {path}")
    return payload


def _head_top_index(category: dict[str, Any]) -> int:
    names = list(category.get("keypoints") or [])
    try:
        return names.index(HEAD_TOP_KEYPOINT_NAME)
    except ValueError as exc:
        raise ValueError(
            f"Category {category.get('id')} does not contain '{HEAD_TOP_KEYPOINT_NAME}'"
        ) from exc


def _to_head_top_coco(source: dict[str, Any]) -> dict[str, Any]:
    categories_by_id = {category["id"]: category for category in source["categories"]}
    kept_annotations = []
    kept_image_ids = set()

    for annotation in source["annotations"]:
        category = categories_by_id.get(annotation.get("category_id"))
        if category is None:
            raise ValueError(
                f"Annotation {annotation.get('id')} references an unknown category"
            )
        index = _head_top_index(category)
        source_keypoints = list(annotation.get("keypoints") or [])
        start = index * 3
        if len(source_keypoints) < start + 3:
            raise ValueError(
                f"Annotation {annotation.get('id')} has no '{HEAD_TOP_KEYPOINT_NAME}' value"
            )
        head_top = source_keypoints[start : start + 3]
        if float(head_top[2]) <= 0:
            continue

        converted = copy.deepcopy(annotation)
        converted["category_id"] = 1
        converted["keypoints"] = head_top
        converted["num_keypoints"] = 1
        kept_annotations.append(converted)
        kept_image_ids.add(annotation["image_id"])

    return {
        "info": copy.deepcopy(source.get("info", {})),
        "licenses": copy.deepcopy(source.get("licenses", [])),
        "images": [
            copy.deepcopy(image)
            for image in source["images"]
            if image["id"] in kept_image_ids
        ],
        "annotations": kept_annotations,
        "categories": [
            {
                "id": 1,
                "name": "infant",
                "supercategory": "person",
                "keypoints": [HEAD_TOP_KEYPOINT_NAME],
                "skeleton": [],
            }
        ],
    }


def prepare_head_top_annotations(dataset_root: Path) -> dict[str, Path]:
    """Derive one-keypoint train/validation COCO annotations from merged labels."""
    annotation_dir = dataset_root.resolve() / "annotations"
    source_names = {
        "train": "person_keypoints_train.json",
        "val": "person_keypoints_val.json",
    }
    output_names = {"train": TRAIN_ANNOTATION_NAME, "val": VAL_ANNOTATION_NAME}
    outputs = {}

    for split, source_name in source_names.items():
        source_path = annotation_dir / source_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Merged annotation not found: {source_path}")
        converted = _to_head_top_coco(_load_coco(source_path))
        if not converted["annotations"]:
            raise ValueError(f"No visible '{HEAD_TOP_KEYPOINT_NAME}' labels in {source_path}")

        output_path = annotation_dir / output_names[split]
        output_path.write_text(
            json.dumps(converted, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        outputs[split] = output_path
    return outputs


def build_head_top_runtime_config(
    dataset_root: Path,
    work_dir: Path,
    base_config: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    device: str,
    checkpoint: Path | None,
    val_interval: int,
    seed: int,
    learning_rate: float | None = None,
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS,
):
    return build_runtime_config(
        base_config=base_config,
        dataset_root=dataset_root,
        work_dir=work_dir,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        checkpoint=checkpoint,
        val_interval=val_interval,
        seed=seed,
        learning_rate=learning_rate,
        warmup_epochs=warmup_epochs,
        keypoint_names=[HEAD_TOP_KEYPOINT_NAME],
        skeleton_links=[],
        dataset_name="infant_head_top",
        train_annotation_name=TRAIN_ANNOTATION_NAME,
        val_annotation_name=VAL_ANNOTATION_NAME,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune RTMPose for infant head_top only."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/infant_dataset_skel/merged"),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("work_dirs/rtmpose_infant_head_top_only"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate the one-keypoint annotations and runtime config without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    work_dir = args.work_dir.resolve()
    prepare_head_top_annotations(dataset_root)
    validate_dataset_root(
        dataset_root,
        train_annotation_name=TRAIN_ANNOTATION_NAME,
        val_annotation_name=VAL_ANNOTATION_NAME,
    )

    base_config = args.base_config.resolve() if args.base_config else find_default_base_config()
    cfg = build_head_top_runtime_config(
        dataset_root=dataset_root,
        work_dir=work_dir,
        base_config=base_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        checkpoint=args.checkpoint,
        val_interval=args.val_interval,
        seed=args.seed,
        learning_rate=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
    )
    config_path = write_runtime_config(cfg, work_dir, "rtmpose_head_top_runtime.py")
    print(f"base_config: {base_config}")
    print(f"runtime_config: {config_path}")
    print(f"dataset_root: {dataset_root}")
    print(f"keypoints: {HEAD_TOP_KEYPOINT_NAME}")
    print(f"epochs: {args.epochs}, batch_size: {args.batch_size}, device: {args.device}")
    if args.dry_run:
        print("dry_run: training was not started")
        return
    run_training(cfg)


if __name__ == "__main__":
    main()
