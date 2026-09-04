import importlib
import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient


class _FakeCardDetector:
    def detect(self, image):
        return {"detected": True, "confidence": 0.95, "px_per_cm": 20.0}


class _FakePoseEstimator:
    def estimate(self, image):
        return {"detected": True, "confidence": 0.87, "keypoints": {}}


class _MissingCheckpointPoseEstimator:
    def estimate(self, image):
        return {
            "detected": False,
            "reason": "head_top_checkpoint_missing",
            "checkpoint": r"C:\secret\models\best_coco_AP_epoch_85.pth",
        }


class _FakeDebugPoseEstimator:
    def estimate(self, image):
        return {
            "detected": True,
            "confidence": 0.87,
            "keypoints": {"head_top": np.array([1.0, 2.0])},
            "keypoint_scores": {"head_top": 0.91},
            "checkpoint": r"C:\secret\models\best_coco_AP_epoch_85.pth",
            "bbox": [0.0, 0.0, 7.0, 7.0],
        }


class _FakeHeightCalculator:
    def calculate(self, keypoints, px_per_cm):
        return {"height_cm": 64.2, "segments": {}}


class _FakeVisualizer:
    def draw_debug(self, image, card_result, pose_result):
        return image


class _NoCardDetector:
    def detect(self, image):
        return {"detected": False, "confidence": 0.0, "px_per_cm": 0.0}


class _BrokenCardDetector:
    def detect(self, image):
        raise RuntimeError("simulated card inference outage")


class JaramHeightWebApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["H_ALIGN_AI_API_KEY"] = "test-ai-key"
        with (
            patch("app.services.card_detector_has.HaSCardDetector"),
            patch("app.services.head_top_pose_estimator.HeadTopPoseEstimator"),
            patch("app.services.height_calculator.HeightCalculator"),
            patch("app.services.visualizer.Visualizer"),
        ):
            main_module = importlib.import_module("app.main")
            cls.measure_module = importlib.import_module("app.routers.measure")
            cls.client = TestClient(main_module.app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("H_ALIGN_AI_API_KEY", None)

    def setUp(self):
        self.original_card_detector = self.measure_module.card_detector
        self.original_pose_estimator = self.measure_module.pose_estimator
        self.original_height_calculator = self.measure_module.height_calculator
        self.original_visualizer = self.measure_module.visualizer
        self.measure_module.card_detector = _FakeCardDetector()
        self.measure_module.pose_estimator = _FakePoseEstimator()
        self.measure_module.height_calculator = _FakeHeightCalculator()
        self.measure_module.visualizer = _FakeVisualizer()

    def tearDown(self):
        os.environ.pop("H_ALIGN_ENABLE_DEBUG_API", None)
        self.measure_module.card_detector = self.original_card_detector
        self.measure_module.pose_estimator = self.original_pose_estimator
        self.measure_module.height_calculator = self.original_height_calculator
        self.measure_module.visualizer = self.original_visualizer

    @staticmethod
    def _image_upload():
        ok, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
        assert ok
        return {"file": ("measurement.png", encoded.tobytes(), "image/png")}

    def test_returns_the_jaram_web_contract_for_an_authenticated_measurement(self):
        response = self.client.post(
            "/api/v1/measure",
            headers={
                "Authorization": "Bearer test-ai-key",
                "X-Measurement-Id": "7d474c14-6208-4380-8c7f-e9609da83f98",
            },
            files=self._image_upload(),
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["measurementId"], "7d474c14-6208-4380-8c7f-e9609da83f98")
        self.assertEqual(body["status"], "SUCCESS")
        self.assertEqual(body["result"]["estimatedHeightCm"], 64.2)
        self.assertIsNone(body["result"]["heightRangeCm"])
        self.assertIsNone(body["result"]["confidence"])
        self.assertIsNone(body["result"]["quality"])
        self.assertNotIn("segmentsCm", body["result"])
        self.assertIsNone(body["error"])

    def test_accepts_matching_non_ascii_bearer_key_without_internal_error(self):
        with patch.dict(os.environ, {"H_ALIGN_AI_API_KEY": "내가 만든 임의의 값"}):
            self.measure_module._verify_ai_credentials("Bearer 내가 만든 임의의 값")

    def test_rejects_mismatched_non_ascii_bearer_key_without_internal_error(self):
        with patch.dict(os.environ, {"H_ALIGN_AI_API_KEY": "내가 만든 임의의 값"}):
            with self.assertRaises(HTTPException) as raised:
                self.measure_module._verify_ai_credentials("Bearer 다른 값")

        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(raised.exception.detail, "Invalid AI server credentials")

    def test_rejects_a_request_without_a_valid_ai_bearer_key(self):
        response = self.client.post(
            "/api/v1/measure",
            headers={"X-Measurement-Id": "7d474c14-6208-4380-8c7f-e9609da83f98"},
            files=self._image_upload(),
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid AI server credentials")

    def test_returns_card_not_found_as_a_retryable_measurement_result(self):
        self.measure_module.card_detector = _NoCardDetector()

        response = self.client.post(
            "/api/v1/measure",
            headers={
                "Authorization": "Bearer test-ai-key",
                "X-Measurement-Id": "7d474c14-6208-4380-8c7f-e9609da83f98",
            },
            files=self._image_upload(),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "measurementId": "7d474c14-6208-4380-8c7f-e9609da83f98",
                "status": "RETRY",
                "result": None,
                "error": {
                    "code": "CARD_NOT_FOUND",
                    "message": "참조 카드를 찾지 못했습니다. 카드가 전체 보이도록 다시 촬영해주세요.",
                },
            },
        )

    def test_returns_missing_checkpoint_as_a_failed_measurement_result_without_paths(self):
        self.measure_module.pose_estimator = _MissingCheckpointPoseEstimator()

        response = self.client.post(
            "/api/v1/measure",
            headers={
                "Authorization": "Bearer test-ai-key",
                "X-Measurement-Id": "7d474c14-6208-4380-8c7f-e9609da83f98",
            },
            files=self._image_upload(),
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["measurementId"], "7d474c14-6208-4380-8c7f-e9609da83f98")
        self.assertEqual(body["status"], "FAILED")
        self.assertEqual(body["error"]["code"], "INFERENCE_FAILED")
        self.assertNotIn("checkpoint", str(body).lower())
        self.assertNotIn(r"C:\secret", str(body))

    def test_returns_http_500_for_an_unexpected_inference_outage(self):
        self.measure_module.card_detector = _BrokenCardDetector()

        response = self.client.post(
            "/api/v1/measure",
            headers={
                "Authorization": "Bearer test-ai-key",
                "X-Measurement-Id": "7d474c14-6208-4380-8c7f-e9609da83f98",
            },
            files=self._image_upload(),
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "AI inference is temporarily unavailable")

    def test_debug_endpoint_is_disabled_by_default(self):
        response = self.client.post(
            "/api/v1/measure/debug",
            files=self._image_upload(),
        )

        self.assertEqual(response.status_code, 404)

    def test_debug_endpoint_does_not_return_local_paths_when_enabled(self):
        os.environ["H_ALIGN_ENABLE_DEBUG_API"] = "1"
        self.measure_module.pose_estimator = _FakeDebugPoseEstimator()

        with patch("app.routers.measure.cv2.imwrite", return_value=True):
            response = self.client.post(
                "/api/v1/measure/debug",
                files=self._image_upload(),
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertNotIn("saved_path", body)
        self.assertNotIn("person_checkpoint", body["ai_analysis"])
        self.assertNotIn(r"C:\secret", str(body))

    def test_openapi_exposes_measure_endpoint_but_not_debug_endpoint(self):
        openapi = self.client.get("/openapi.json").json()
        paths = openapi["paths"]

        self.assertIn("/api/v1/measure", paths)
        self.assertNotIn("/api/v1/measure/debug", paths)


if __name__ == "__main__":
    unittest.main()
