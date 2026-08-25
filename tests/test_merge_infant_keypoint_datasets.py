import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "merge_infant_keypoint_datasets.py"
)
spec = importlib.util.spec_from_file_location("merge_infant_keypoint_datasets", SCRIPT_PATH)
merge_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge_module)


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


def make_category(keypoints=None):
    return {
        "id": 1,
        "name": "infant",
        "supercategory": "person",
        "keypoints": keypoints or CANONICAL_KEYPOINTS,
        "skeleton": [[1, 2], [2, 4]],
    }


def make_annotation(annotation_id, image_id, keypoints):
    visible_count = sum(
        1 for index in range(2, len(keypoints), 3) if keypoints[index] > 0
    )
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [0, 0, 100, 100],
        "area": 10000,
        "iscrowd": 0,
        "keypoints": keypoints,
        "num_keypoints": visible_count,
    }


def write_source(root, name, image_entries, category=None):
    source = root / name
    image_root = source / "images"
    annotation_root = source / "annotations"
    image_root.mkdir(parents=True)
    annotation_root.mkdir()

    images = []
    annotations = []
    for image_id, file_name, image_bytes, keypoints in image_entries:
        image_path = source / file_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)
        images.append({"id": image_id, "file_name": file_name, "width": 100, "height": 100})
        annotations.append(make_annotation(image_id, image_id, keypoints))

    payload = {
        "info": {"description": "test"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [category or make_category()],
    }
    (annotation_root / "person_keypoints_train.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return source


class MergeInfantKeypointDatasetsTests(unittest.TestCase):
    def test_discover_source_dirs_ignores_zip_and_non_matching_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            splits = Path(tmp) / "splits"
            splits.mkdir()
            source = splits / "명준_infant_dataset_skel_part01_0001-0002"
            source.mkdir()
            (splits / "윤민_infant_dataset_skel_part02_0003-0004.zip").write_bytes(b"zip")
            (splits / "notes").mkdir()

            result = merge_module.discover_source_dirs(splits)

        self.assertEqual(result, [source])

    def test_normalize_annotation_reorders_keypoints_by_category_name(self):
        source_order = ["head_top", "nose"] + CANONICAL_KEYPOINTS[1:-1]
        source_order = list(dict.fromkeys(source_order))
        source_keypoints = []
        for index in range(len(source_order)):
            source_keypoints.extend([index + 30, index + 40, 2])
        annotation = {
            "id": 1,
            "image_id": 2,
            "category_id": 1,
            "keypoints": source_keypoints,
        }

        normalized = merge_module.normalize_annotation(
            annotation,
            {"keypoints": source_order},
            image_id=7,
            annotation_id=9,
        )

        self.assertEqual(normalized["image_id"], 7)
        self.assertEqual(normalized["id"], 9)
        self.assertEqual(normalized["keypoints"][:6], [31, 41, 2, 32, 42, 2])

    def test_merge_rewrites_ids_copies_images_and_splits_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits = root / "splits"
            splits.mkdir()
            visible = []
            for index in range(14):
                visible.extend([float(index + 1), float(index + 2), 2])
            first = write_source(
                splits,
                "명준_infant_dataset_skel_part01_0001-0002",
                [(10, "images/train/infant_train_000001.jpg", b"one", visible)],
            )
            second = write_source(
                splits,
                "윤민_infant_dataset_skel_part02_0003-0004",
                [(20, "images/default/images/train/infant_train_000003.jpg", b"two", visible)],
            )
            output = root / "merged"

            manifest = merge_module.merge_datasets(splits, output, val_ratio=0.5, seed=42)

            merged = json.loads(
                (output / "annotations" / "person_keypoints.json").read_text(encoding="utf-8")
            )
            train = json.loads(
                (output / "annotations" / "person_keypoints_train.json").read_text(encoding="utf-8")
            )
            val = json.loads(
                (output / "annotations" / "person_keypoints_val.json").read_text(encoding="utf-8")
            )

            self.assertEqual(manifest["source_count"], 2)
            self.assertEqual(len(merged["images"]), 2)
            self.assertEqual([image["id"] for image in merged["images"]], [1, 2])
            self.assertEqual([annotation["id"] for annotation in merged["annotations"]], [1, 2])
            self.assertEqual(len(train["images"]), 1)
            self.assertEqual(len(val["images"]), 1)
            self.assertEqual(
                {image["id"] for image in train["images"]}
                | {image["id"] for image in val["images"]},
                {1, 2},
            )
            self.assertEqual(len(list((output / "images").glob("*.jpg"))), 2)
            for image in merged["images"]:
                self.assertEqual(Path(image["file_name"]).parent, Path("."))
                self.assertTrue((output / "images" / image["file_name"]).is_file())
            self.assertEqual(
                {image["source_dataset"] for image in merged["images"]},
                {first.name, second.name},
            )
            self.assertEqual(manifest["missing_keypoint_counts"], {})

            repeat_output = root / "merged-repeat"
            merge_module.merge_datasets(splits, repeat_output, val_ratio=0.5, seed=42)
            repeat_train = json.loads(
                (repeat_output / "annotations" / "person_keypoints_train.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                [image["id"] for image in train["images"]],
                [image["id"] for image in repeat_train["images"]],
            )


if __name__ == "__main__":
    unittest.main()
