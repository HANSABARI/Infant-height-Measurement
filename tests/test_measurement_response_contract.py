import unittest

from app.services.measurement_response import build_error_response, build_success_response


class MeasurementResponseContractTests(unittest.TestCase):
    def test_build_success_response_returns_data_payload_for_backend_ai_contract(self):
        response = build_success_response(
            baby_id="baby_123",
            calc_result={
                "height_cm": 64.2,
                "segments": {
                    "head": 13.2,
                    "torso": 24.1,
                    "leg": 26.9,
                },
            },
            card_result={"confidence": 0.95},
            pose_result={"confidence": 0.87},
        )

        self.assertEqual(
            response,
            {
                "success": True,
                "data": {
                    "baby_id": "baby_123",
                    "height_cm": 64.2,
                    "height_range": {
                        "min": 63.7,
                        "max": 64.7,
                    },
                    "confidence": 0.91,
                    "method": "h-align-v1",
                    "segments_cm": {
                        "head": 13.2,
                        "torso": 24.1,
                        "leg": 26.9,
                    },
                    "knee_angle": 180.0,
                    "warnings": [],
                },
                "error": None,
            },
        )

    def test_build_error_response_returns_structured_error_code(self):
        response = build_error_response(
            code="CARD_NOT_FOUND",
            message="참조 카드를 찾을 수 없습니다. 카드가 잘 보이게 찍어주세요.",
        )

        self.assertEqual(
            response,
            {
                "success": False,
                "data": None,
                "error": {
                    "code": "CARD_NOT_FOUND",
                    "message": "참조 카드를 찾을 수 없습니다. 카드가 잘 보이게 찍어주세요.",
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
