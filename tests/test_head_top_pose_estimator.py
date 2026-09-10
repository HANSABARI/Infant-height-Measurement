import unittest
import tempfile
import importlib.util
import sys
import types
from unittest.mock import patch
from pathlib import Path
from runpy import run_path

import numpy as np

if importlib.util.find_spec("rtmlib") is None:
    sys.modules["rtmlib"] = types.SimpleNamespace(Wholebody=lambda *args, **kwargs: None)

from app.services.head_top_pose_estimator import (
    HEAD_TOP_KEYPOINT_NAME,
    WHOLEBODY_MEASUREMENT_KEYPOINT_NAMES,
    HeadTopPoseEstimator,
)


class _FakePersonProposer:
    def estimate(self, image):
        return {
            "detected": True,
            "confidence": 0.8,
            "bbox": [20.0, 40.0, 80.0, 160.0],
            "keypoints": {
                name: np.array([index + 10.0, index + 20.0])
                for index, name in enumerate(WHOLEBODY_MEASUREMENT_KEYPOINT_NAMES)
            },
        }


class _FakePredInstances:
    def __init__(self, keypoints, keypoint_scores):
        self.keypoints = keypoints
        self.keypoint_scores = keypoint_scores


class _FakeDataSample:
    def __init__(self, point, score):
        self.pred_instances = _FakePredInstances(point, score)


class _CapturingPoseEstimator:
    def __init__(self, device):
        self.device = device


class HeadTopPoseEstimatorTests(unittest.TestCase):
    def test_uses_the_packaged_runtime_config_by_default(self):
        estimator = HeadTopPoseEstimator(
            model=object(),
            person_proposer=_FakePersonProposer(),
            inference_fn=lambda *args, **kwargs: [],
        )

        project_root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            estimator.config_path,
            project_root / "app" / "configs" / "rtmpose_head_top.py",
        )

    def test_uses_packaged_head_top_checkpoint_dir_by_default(self):
        estimator = HeadTopPoseEstimator(
            model=object(),
            person_proposer=_FakePersonProposer(),
            inference_fn=lambda *args, **kwargs: [],
        )

        project_root = Path(__file__).resolve().parents[1]
        expected_dir = project_root / "app" / "models" / "rtmpose_infant_head_top"
        self.assertEqual(estimator.default_work_dir, expected_dir)
        self.assertEqual(estimator.checkpoint_path.parent, expected_dir)

    def test_default_device_prefers_cuda_when_available(self):
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: False)
            ),
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(HeadTopPoseEstimator._default_device(), "cuda")

    def test_default_person_proposer_uses_selected_device(self):
        with patch(
            "app.services.head_top_pose_estimator.PoseEstimator",
            _CapturingPoseEstimator,
        ):
            estimator = HeadTopPoseEstimator(
                device="cuda",
                model=object(),
                inference_fn=lambda *args, **kwargs: [],
            )

        self.assertEqual(estimator.person_proposer.device, "cuda")

    def test_packaged_runtime_config_keeps_the_inference_pipeline(self):
        project_root = Path(__file__).resolve().parents[1]
        config = run_path(project_root / "app" / "configs" / "rtmpose_head_top.py")

        pipeline = config["test_dataloader"]["dataset"]["pipeline"]
        self.assertEqual(
            [step["type"] for step in pipeline],
            ["LoadImage", "GetBBoxCenterScale", "TopdownAffine", "PackPoseInputs"],
        )

    def test_finds_saved_best_checkpoint_without_assuming_epoch_100(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            expected = work_dir / "best_coco_AP_epoch_17.pth"
            expected.touch()

            checkpoint = HeadTopPoseEstimator.find_best_checkpoint(work_dir)

        self.assertEqual(checkpoint, expected)

    def test_defers_missing_checkpoint_failure_until_inference(self):
        estimator = HeadTopPoseEstimator(
            config_path="/tmp/missing-head-top-config.py",
            checkpoint_path="/tmp/missing-head-top-checkpoint.pth",
            person_proposer=_FakePersonProposer(),
            inference_fn=lambda *args, **kwargs: [],
        )

        result = estimator.estimate(np.zeros((200, 100, 3), dtype=np.uint8))

        self.assertFalse(result["detected"])
        self.assertEqual(result["reason"], "head_top_checkpoint_missing")

    def test_keeps_wholebody_keypoints_and_adds_head_top(self):
        captured = {}

        def fake_inference(model, image, bboxes):
            captured["bboxes"] = bboxes
            return [
                _FakeDataSample(
                    np.asarray([[[44.0, 55.0]]], dtype=np.float32),
                    np.asarray([[0.75]], dtype=np.float32),
                )
            ]

        estimator = HeadTopPoseEstimator(
            model=object(),
            person_proposer=_FakePersonProposer(),
            inference_fn=fake_inference,
        )
        result = estimator.estimate(np.zeros((200, 100, 3), dtype=np.uint8))

        self.assertTrue(result["detected"])
        self.assertEqual(
            list(result["keypoints"]),
            [*WHOLEBODY_MEASUREMENT_KEYPOINT_NAMES, HEAD_TOP_KEYPOINT_NAME],
        )
        np.testing.assert_array_equal(result["keypoints"]["nose"], [10.0, 20.0])
        np.testing.assert_array_equal(result["keypoints"]["head_top"], [44.0, 55.0])
        self.assertEqual(result["bbox"], [14.0, 28.0, 86.0, 172.0])
        np.testing.assert_array_equal(captured["bboxes"], [[14.0, 28.0, 86.0, 172.0]])

    def test_rejects_low_confidence_head_top(self):
        def fake_inference(model, image, bboxes):
            return [
                _FakeDataSample(
                    np.asarray([[[44.0, 55.0]]], dtype=np.float32),
                    np.asarray([[0.05]], dtype=np.float32),
                )
            ]

        estimator = HeadTopPoseEstimator(
            model=object(),
            person_proposer=_FakePersonProposer(),
            inference_fn=fake_inference,
        )
        result = estimator.estimate(np.zeros((200, 100, 3), dtype=np.uint8))

        self.assertFalse(result["detected"])
        self.assertEqual(result["reason"], "head_top_low_confidence")


if __name__ == "__main__":
    unittest.main()
