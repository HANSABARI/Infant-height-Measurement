import unittest
import tempfile
import os
import sys
import types
from unittest.mock import patch
from pathlib import Path

import numpy as np

from app.services.pose_estimator import PoseEstimator


class _FakeWholebody:
    def __init__(self, mode, backend, device):
        self.mode = mode
        self.backend = backend
        self.device = device


class PoseEstimatorTest(unittest.TestCase):
    def test_uses_onnxruntime_default_preload_order_for_cuda_device(self):
        preload_calls = []
        fake_onnxruntime = types.SimpleNamespace(
            preload_dlls=lambda **kwargs: preload_calls.append(kwargs)
        )

        with (
            patch.dict(sys.modules, {"onnxruntime": fake_onnxruntime}),
            patch("app.services.pose_estimator.Wholebody", _FakeWholebody),
        ):
            estimator = PoseEstimator(device="cuda")

        self.assertEqual(preload_calls, [{}])
        self.assertEqual(estimator.pose.device, "cuda")

    def test_adds_pytorch_dll_directory_for_cuda_device(self):
        added_directories = []
        fake_onnxruntime = types.SimpleNamespace(preload_dlls=lambda **kwargs: None)

        with tempfile.TemporaryDirectory() as tmp:
            site_packages = Path(tmp)
            torch_package = site_packages / "torch"
            torch_lib = torch_package / "lib"
            torch_lib.mkdir(parents=True)
            fake_torch = types.SimpleNamespace(__file__=str(torch_package / "__init__.py"))

            with (
                patch.dict(
                    sys.modules,
                    {"onnxruntime": fake_onnxruntime, "torch": fake_torch},
                ),
                patch("app.services.pose_estimator.os.name", "nt"),
                patch.dict("os.environ", {"PATH": "C:\\Windows\\System32"}),
                patch(
                    "os.add_dll_directory",
                    lambda path: added_directories.append(path) or object(),
                    create=True,
                ),
                patch("app.services.pose_estimator.Wholebody", _FakeWholebody),
            ):
                PoseEstimator(device="cuda")
                path_after_preload = os.environ["PATH"]

        self.assertEqual(added_directories, [str(torch_lib)])
        self.assertTrue(path_after_preload.startswith(f"{torch_lib};"))

    def test_builds_person_bbox_from_confident_keypoints_only(self):
        keypoints = np.array(
            [[10.0, 20.0], [90.0, 180.0], [999.0, 999.0]],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8, 0.01], dtype=np.float32)

        bbox = PoseEstimator._bbox_from_scored_keypoints(keypoints, scores)

        self.assertEqual(bbox, [10.0, 20.0, 90.0, 180.0])


if __name__ == "__main__":
    unittest.main()
