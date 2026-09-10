import unittest
from datetime import datetime, timezone

from app.services.jaram_height_web_response import (
    build_failed_response,
    build_retry_response,
    build_success_response,
)
from app.schemas.jaram_height_web import (
    WebMeasurementFailedResponse,
    WebMeasurementRetryResponse,
    WebMeasurementSuccessResponse,
)


class JaramHeightWebResponseTests(unittest.TestCase):
    def test_builds_the_web_ai_success_contract_without_internal_diagnostics(self):
        response = build_success_response(
            measurement_id="7d474c14-6208-4380-8c7f-e9609da83f98",
            calc_result={
                "height_cm": 64.2,
                "segments": {"head": 13.2, "torso": 24.1, "leg": 26.9},
            },
            measured_at=datetime(2026, 8, 23, 12, 30, tzinfo=timezone.utc),
        )

        self.assertEqual(
            response,
            {
                "measurementId": "7d474c14-6208-4380-8c7f-e9609da83f98",
                "status": "SUCCESS",
                "result": {
                    "estimatedHeightCm": 64.2,
                    "heightRangeCm": None,
                    "confidence": None,
                    "quality": None,
                    "modelVersion": "h-align-has-rtmpose-head-top-v1",
                    "warnings": [],
                    "measuredAt": "2026-08-23T12:30:00Z",
                },
                "error": None,
            },
        )

    def test_leaves_uncalibrated_measurement_metadata_null(self):
        response = build_success_response(
            measurement_id="2fc5929f-a8d6-4011-8762-4f4d0056b6b6",
            calc_result={"height_cm": 64.2, "segments": {}},
            measured_at=datetime(2026, 8, 23, tzinfo=timezone.utc),
        )

        self.assertIsNone(response["result"]["heightRangeCm"])
        self.assertIsNone(response["result"]["confidence"])
        self.assertIsNone(response["result"]["quality"])

    def test_builds_a_contract_valid_retry_response(self):
        response = build_retry_response(
            measurement_id="7d474c14-6208-4380-8c7f-e9609da83f98",
            code="CARD_NOT_FOUND",
            message="참조 카드를 찾지 못했습니다.",
        )

        self.assertEqual(
            response,
            {
                "measurementId": "7d474c14-6208-4380-8c7f-e9609da83f98",
                "status": "RETRY",
                "result": None,
                "error": {
                    "code": "CARD_NOT_FOUND",
                    "message": "참조 카드를 찾지 못했습니다.",
                },
            },
        )

    def test_builds_a_contract_valid_failed_response_without_internals(self):
        response = build_failed_response(
            measurement_id="7d474c14-6208-4380-8c7f-e9609da83f98",
            message="AI 추론 모델을 준비하지 못했습니다.",
        )

        self.assertEqual(
            response,
            {
                "measurementId": "7d474c14-6208-4380-8c7f-e9609da83f98",
                "status": "FAILED",
                "result": None,
                "error": {
                    "code": "INFERENCE_FAILED",
                    "message": "AI 추론 모델을 준비하지 못했습니다.",
                },
            },
        )
        self.assertNotIn("checkpoint", str(response).lower())
        self.assertNotIn("traceback", str(response).lower())

    def test_swagger_examples_match_the_public_response_contract(self):
        success_example = WebMeasurementSuccessResponse.model_json_schema()["examples"][0]
        retry_example = WebMeasurementRetryResponse.model_json_schema()["examples"][0]
        failed_example = WebMeasurementFailedResponse.model_json_schema()["examples"][0]

        self.assertEqual(
            success_example,
            {
                "measurementId": "7d474c14-6208-4380-8c7f-e9609da83f98",
                "status": "SUCCESS",
                "result": {
                    "estimatedHeightCm": 64.2,
                    "heightRangeCm": None,
                    "confidence": None,
                    "quality": None,
                    "modelVersion": "h-align-has-rtmpose-head-top-v1",
                    "warnings": [],
                    "measuredAt": "2026-08-23T12:30:00Z",
                },
                "error": None,
            },
        )
        self.assertEqual(
            retry_example,
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
        self.assertEqual(
            failed_example,
            {
                "measurementId": "7d474c14-6208-4380-8c7f-e9609da83f98",
                "status": "FAILED",
                "result": None,
                "error": {
                    "code": "INFERENCE_FAILED",
                    "message": "AI 추론 모델을 준비하지 못했습니다.",
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
