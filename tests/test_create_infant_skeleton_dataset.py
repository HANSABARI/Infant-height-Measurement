import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "create_infant_skeleton_dataset.py"
)

spec = importlib.util.spec_from_file_location("create_infant_skeleton_dataset", SCRIPT_PATH)
skel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skel)


class CreateInfantSkeletonDatasetTests(unittest.TestCase):
    def test_iter_image_files_recurses_and_sorts_supported_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.jpg").write_bytes(b"")
            (root / "nested").mkdir()
            (root / "nested" / "a.png").write_bytes(b"")
            (root / "notes.txt").write_text("ignore me")

            result = [path.relative_to(root).as_posix() for path in skel.iter_image_files(root)]

        self.assertEqual(result, ["b.jpg", "nested/a.png"])

    def test_normalize_pose_result_keeps_named_keypoints_and_scores(self):
        pose_result = {
            "detected": True,
            "keypoints": {
                "pt_0": [10.0, 20.0],
                "nose": [10.0, 20.0],
                "left_shoulder": [8.0, 40.0],
            },
            "scores": {"pt_0": 0.9, "nose": 0.9, "left_shoulder": 0.7},
            "confidence": 0.8,
        }

        normalized = skel.normalize_pose_result(pose_result, ["nose", "left_shoulder", "head_top"])

        self.assertEqual(normalized["keypoints"], [[10.0, 20.0], [8.0, 40.0], [0.0, 0.0]])
        self.assertEqual(normalized["scores"], [0.9, 0.7, 0.0])

    def test_build_named_pose_result_uses_pose_estimator_update_indices(self):
        keypoints = [[[float(i), float(i + 100)] for i in range(133)]]
        scores = [[float(i) / 100.0 for i in range(133)]]

        result = skel.build_named_pose_result(keypoints, scores)

        self.assertTrue(result["detected"])
        self.assertEqual(result["keypoints"]["nose"], [0.0, 100.0])
        self.assertEqual(result["keypoints"]["left_shoulder"], [5.0, 105.0])
        self.assertEqual(result["keypoints"]["right_heel"], [22.0, 122.0])
        self.assertEqual(result["scores"]["nose"], 0.0)
        self.assertEqual(result["scores"]["left_shoulder"], 0.05)
        self.assertEqual(result["scores"]["right_heel"], 0.22)

    def test_build_coco_annotation_marks_missing_head_top_as_unlabeled(self):
        normalized = {
            "keypoints": [[10.0, 20.0], [8.0, 40.0], [0.0, 0.0]],
            "scores": [0.9, 0.7, 0.0],
            "confidence": 0.8,
        }

        annotation = skel.build_coco_annotation(
            annotation_id=1,
            image_id=10,
            normalized=normalized,
            score_threshold=0.3,
        )

        self.assertEqual(annotation["keypoints"], [10.0, 20.0, 2, 8.0, 40.0, 2, 0.0, 0.0, 0])
        self.assertEqual(annotation["num_keypoints"], 2)
        self.assertEqual(annotation["bbox"], [8.0, 20.0, 2.0, 20.0])

    def test_build_cvat_coco_keypoints_archive_uses_expected_zip_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "train"
            image_dir = split_dir / "images"
            image_dir.mkdir(parents=True)
            (image_dir / "sample.jpg").write_bytes(b"fake image")
            annotations = {
                "images": [
                    {"id": 1, "file_name": "images/sample.jpg", "width": 100, "height": 50}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "iscrowd": 0,
                        "bbox": [1, 2, 3, 4],
                        "area": 12,
                        "num_keypoints": 1,
                        "keypoints": [1, 2, 2, 0, 0, 0],
                    }
                ],
                "categories": [
                    {
                        "id": 1,
                        "name": "infant",
                        "supercategory": "person",
                        "keypoints": ["nose", "head_top"],
                        "skeleton": [[1, 2]],
                    }
                ],
            }
            (split_dir / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")
            zip_path = root / "cvat.zip"

            skel.build_cvat_coco_keypoints_archive(split_dir, zip_path, "train")

            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(
                    sorted(archive.namelist()),
                    ["annotations/person_keypoints_train.json", "images/train/sample.jpg"],
                )
                cvat_json = json.loads(
                    archive.read("annotations/person_keypoints_train.json").decode("utf-8")
                )

        self.assertEqual(cvat_json["images"][0]["file_name"], "train/sample.jpg")
        self.assertEqual(cvat_json["categories"][0]["keypoints"], ["nose", "head_top"])
        self.assertEqual(cvat_json["annotations"][0]["keypoints"][-3:], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
