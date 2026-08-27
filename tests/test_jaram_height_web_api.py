import importlib
import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient


class _FakeCardDetector:
    def detect(self, image):
        return {"detected": True, "confidence": 0.95, "px_per_cm": 20.0}


class _FakePoseEstimator:
    def estimate(self, image):
        return {"detected": True, "confidence": 0.87, "keypoints": {}}


class _FakeHeightCalculator:
    def calculate(self, keypoints, px_per_cm):
        return {"height_cm": 64.2, "segments": {}}


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
        os.environ.pop("H_ALIGN_DEBUG_API_ENABLED", None)
        self.original_card_detector = self.measure_module.card_detector
        self.original_pose_estimator = self.measure_module.pose_estimator
        self.original_height_calculator = self.measure_module.height_calculator
        self.measure_module.card_detector = _FakeCardDetector()
        self.measure_module.pose_estimator = _FakePoseEstimator()
        self.measure_module.height_calculator = _FakeHeightCalculator()

    def tearDown(self):
        os.environ.pop("H_ALIGN_DEBUG_API_ENABLED", None)
        self.measure_module.card_detector = self.original_card_detector
        self.measure_module.pose_estimator = self.original_pose_estimator
        self.measure_module.height_calculator = self.original_height_calculator

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
            headers={"Authorization": "Bearer test-ai-key"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Not Found")

    def test_debug_endpoint_requires_authentication_when_enabled(self):
        os.environ["H_ALIGN_DEBUG_API_ENABLED"] = "1"

        response = self.client.post(
            "/api/v1/measure/debug",
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid AI server credentials")

    def test_debug_endpoint_runs_when_explicitly_enabled_and_authenticated(self):
        os.environ["H_ALIGN_DEBUG_API_ENABLED"] = "true"
        debug_image = np.zeros((8, 8, 3), dtype=np.uint8)

        with (
            patch.object(self.measure_module.visualizer, "draw_debug", return_value=debug_image),
            patch("app.routers.measure.os.makedirs") as makedirs,
            patch("app.routers.measure.cv2.imwrite", return_value=True) as imwrite,
        ):
            response = self.client.post(
                "/api/v1/measure/debug",
                headers={"Authorization": "Bearer test-ai-key"},
                files=self._image_upload(),
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        makedirs.assert_called_once_with(self.measure_module.DEBUG_DIR, exist_ok=True)
        imwrite.assert_called_once()


if __name__ == "__main__":
    unittest.main()
