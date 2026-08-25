from datetime import datetime, timezone
import os
from unittest import TestCase

from fastapi.testclient import TestClient

import app.routers.measure as measure_router
from app.main import app
from app.schemas.measurement import MeasurementError, MeasurementResponse, MeasurementResult


class MeasurementContractTests(TestCase):
    def test_success_allows_uncalibrated_metrics_to_be_null(self) -> None:
        response = MeasurementResponse(
            measurementId="measurement-1",
            status="SUCCESS",
            result=MeasurementResult(
                estimatedHeightCm=64.2,
                heightRangeCm=None,
                confidence=None,
                quality=None,
                modelVersion="h-align-v1",
                warnings=["HEIGHT_RANGE_UNCALIBRATED", "CONFIDENCE_UNCALIBRATED"],
                measuredAt=datetime.now(timezone.utc),
            ),
            error=None,
        )

        self.assertIsNone(response.result.heightRangeCm)
        self.assertIsNone(response.result.confidence)
        self.assertIsNone(response.result.quality)
        self.assertEqual(response.result.modelVersion, "h-align-v1")

    def test_retry_and_failed_responses_have_null_results(self) -> None:
        for status, code in (("RETRY", "CARD_NOT_FOUND"), ("FAILED", "INFERENCE_FAILED")):
            response = MeasurementResponse(
                measurementId="measurement-1",
                status=status,
                result=None,
                error=MeasurementError(code=code, message="error"),
            )

            self.assertIsNone(response.result)
            self.assertEqual(response.error.code, code)

    def test_measure_route_adapts_success_without_uncalibrated_values(self) -> None:
        original_models = measure_router._models
        original_decode = measure_router._decode_image
        previous_key = os.environ.get("AI_API_KEY")

        class FakeCardDetector:
            def detect(self, _image: object) -> dict[str, object]:
                return {"detected": True, "px_per_cm": 10.0, "confidence": 0.9}

        class FakePoseEstimator:
            def estimate(self, _image: object) -> dict[str, object]:
                return {"detected": True, "keypoints": {}, "confidence": 0.8}

        class FakeHeightCalculator:
            def calculate(self, keypoints: dict[str, object], px_per_cm: float) -> dict[str, object]:
                return {"height_cm": 64.2, "segments": {}}

        try:
            os.environ["AI_API_KEY"] = "test-key"
            measure_router._models = (
                FakeCardDetector(),
                FakePoseEstimator(),
                FakeHeightCalculator(),
                object(),
            )
            measure_router._decode_image = lambda _contents: object()

            response = TestClient(app).post(
                "/api/v1/measure",
                headers={
                    "Authorization": "Bearer test-key",
                    "X-Measurement-Id": "measurement-1",
                },
                files={"file": ("photo.jpg", b"image", "image/jpeg")},
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {
                "measurementId": "measurement-1",
                "status": "SUCCESS",
                "result": {
                    "estimatedHeightCm": 64.2,
                    "heightRangeCm": None,
                    "confidence": None,
                    "quality": None,
                    "modelVersion": "h-align-v1",
                    "warnings": [
                        "HEIGHT_RANGE_UNCALIBRATED",
                        "CONFIDENCE_UNCALIBRATED",
                    ],
                    "measuredAt": response.json()["result"]["measuredAt"],
                },
                "error": None,
            })
        finally:
            measure_router._models = original_models
            measure_router._decode_image = original_decode
            if previous_key is None:
                os.environ.pop("AI_API_KEY", None)
            else:
                os.environ["AI_API_KEY"] = previous_key
