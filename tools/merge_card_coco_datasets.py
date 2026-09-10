import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from random import Random


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif")


def _image_candidates(dataset_root: Path, file_name: str) -> list[Path]:
    path = Path(file_name)
    stems = [path]

    if path.suffix:
        stems.extend(path.with_suffix(ext) for ext in IMAGE_EXTS)
        stems.extend(path.with_suffix(ext.upper()) for ext in IMAGE_EXTS)
    else:
        stems.extend(Path(file_name + ext) for ext in IMAGE_EXTS)
        stems.extend(Path(file_name + ext.upper()) for ext in IMAGE_EXTS)

    candidates: list[Path] = []
    for candidate in dict.fromkeys(stems):
        candidates.extend(
            [
                dataset_root / "images" / candidate,
                dataset_root / "images" / "default" / candidate,
                dataset_root / candidate,
            ]
        )

    return candidates


def _find_image(dataset_root: Path, file_name: str) -> Path:
    for candidate in _image_candidates(dataset_root, file_name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image for '{file_name}' not found under {dataset_root}")


def _safe_prefix(path: Path) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in path.name).strip("_").lower()


def _load_coco(dataset_root: Path) -> dict:
    annotation_path = dataset_root / "annotations" / "instances_default.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _subset(coco: dict, image_ids: set[int]) -> dict:
    data = deepcopy(coco)
    data["images"] = [image for image in coco["images"] if image["id"] in image_ids]
    data["annotations"] = [
        annotation for annotation in coco["annotations"] if annotation["image_id"] in image_ids
    ]
    return data


def merge_datasets(dataset_roots: list[Path], output_root: Path, val_ratio: float, seed: int) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")

    image_output = output_root / "images" / "default"
    annotation_output = output_root / "annotations"
    image_output.mkdir(parents=True, exist_ok=True)
    annotation_output.mkdir(parents=True, exist_ok=True)

    merged = {
        "info": {"description": "Merged card instance dataset"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "card", "supercategory": ""}],
    }

    next_image_id = 1
    next_annotation_id = 1
    source_summaries = []

    for dataset_root in dataset_roots:
        dataset_root = dataset_root.resolve()
        coco = _load_coco(dataset_root)
        prefix = _safe_prefix(dataset_root)
        id_map: dict[int, int] = {}

        for image in coco.get("images", []):
            source_image = _find_image(dataset_root, image["file_name"])
            output_name = f"{prefix}__{source_image.name}"
            destination = image_output / output_name
            shutil.copy2(source_image, destination)

            new_image = deepcopy(image)
            old_image_id = image["id"]
            id_map[old_image_id] = next_image_id
            new_image["id"] = next_image_id
            new_image["file_name"] = output_name
            new_image["source_dataset"] = dataset_root.name
            merged["images"].append(new_image)
            next_image_id += 1

        annotation_count = 0
        for annotation in coco.get("annotations", []):
            old_image_id = annotation["image_id"]
            if old_image_id not in id_map:
                continue

            new_annotation = deepcopy(annotation)
            new_annotation["id"] = next_annotation_id
            new_annotation["image_id"] = id_map[old_image_id]
            new_annotation["category_id"] = 1
            merged["annotations"].append(new_annotation)
            next_annotation_id += 1
            annotation_count += 1

        source_summaries.append(
            {
                "name": dataset_root.name,
                "images": len(coco.get("images", [])),
                "annotations": annotation_count,
            }
        )

    rng = Random(seed)
    image_ids = [image["id"] for image in merged["images"]]
    rng.shuffle(image_ids)
    val_count = max(1, round(len(image_ids) * val_ratio))
    val_ids = set(image_ids[:val_count])
    train_ids = set(image_ids[val_count:])

    outputs = {
        "instances_default.json": merged,
        "instances_train.json": _subset(merged, train_ids),
        "instances_val.json": _subset(merged, val_ids),
    }
    for name, data in outputs.items():
        with (annotation_output / name).open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)

    print(f"output: {output_root}")
    for summary in source_summaries:
        print(f"{summary['name']}: images={summary['images']} annotations={summary['annotations']}")
    print(f"merged: images={len(merged['images'])} annotations={len(merged['annotations'])}")
    print(f"train: images={len(train_ids)} annotations={len(outputs['instances_train.json']['annotations'])}")
    print(f"val: images={len(val_ids)} annotations={len(outputs['instances_val.json']['annotations'])}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge card COCO datasets into one dataset root.")
    parser.add_argument("dataset_roots", nargs="+", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    merge_datasets(args.dataset_roots, args.output_root, args.val_ratio, args.seed)
