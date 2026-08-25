import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_rtmpose_head_top.py"
spec = importlib.util.spec_from_file_location("train_rtmpose_head_top", SCRIPT_PATH)
head_top_train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(head_top_train_module)


class TrainRtmposeHeadTopTests(unittest.TestCase):
    def test_prepares_one_keypoint_annotations_and_omits_unlabeled_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp) / "merged"
            annotation_dir = dataset_root / "annotations"
            annotation_dir.mkdir(parents=True)
            category = {
                "id": 1,
                "name": "infant",
                "keypoints": ["nose", "head_top"],
                "skeleton": [],
            }
            source = {
                "images": [
                    {"id": 1, "file_name": "one.jpg"},
                    {"id": 2, "file_name": "two.jpg"},
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 10, 10],
                        "keypoints": [1, 2, 2, 3, 4, 2],
                        "num_keypoints": 2,
                    },
                    {
                        "id": 2,
                        "image_id": 2,
                        "category_id": 1,
                        "bbox": [0, 0, 10, 10],
                        "keypoints": [5, 6, 2, 0, 0, 0],
                        "num_keypoints": 1,
                    },
                ],
                "categories": [category],
            }
            for split in ("train", "val"):
                (annotation_dir / f"person_keypoints_{split}.json").write_text(
                    json.dumps(source), encoding="utf-8"
                )

            outputs = head_top_train_module.prepare_head_top_annotations(dataset_root)
            train = json.loads(outputs["train"].read_text(encoding="utf-8"))

        self.assertEqual(train["categories"][0]["keypoints"], ["head_top"])
        self.assertEqual([image["id"] for image in train["images"]], [1])
        self.assertEqual(len(train["annotations"]), 1)
        self.assertEqual(train["annotations"][0]["keypoints"], [3, 4, 2])
        self.assertEqual(train["annotations"][0]["num_keypoints"], 1)

    def test_builds_one_keypoint_runtime_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation_dir = root / "merged" / "annotations"
            annotation_dir.mkdir(parents=True)
            payload = {"images": [{"id": index} for index in range(8)]}
            for split in ("train", "val"):
                (annotation_dir / f"person_keypoints_head_top_{split}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            cfg = head_top_train_module.build_head_top_runtime_config(
                dataset_root=root / "merged",
                work_dir=root / "work",
                base_config=head_top_train_module.find_default_base_config(),
                epochs=10,
                batch_size=4,
                num_workers=0,
                device="cpu",
                checkpoint=None,
                val_interval=2,
                seed=42,
            )

        self.assertEqual(cfg.model.head.out_channels, 1)
        self.assertEqual(cfg.train_dataloader.dataset.ann_file, "annotations/person_keypoints_head_top_train.json")
        self.assertEqual(cfg.val_dataloader.dataset.ann_file, "annotations/person_keypoints_head_top_val.json")
        self.assertEqual(cfg.train_dataloader.dataset.metainfo["keypoint_info"][0]["name"], "head_top")
