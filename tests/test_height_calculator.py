import unittest

import numpy as np

from app.services.height_calculator import HeightCalculator


class HeightCalculatorTest(unittest.TestCase):
    def test_calculates_height_from_actual_head_top_and_heel(self):
        keypoints = {
            "nose": np.array([1.0, 1.0]),
            "head_top": np.array([1.0, 0.0]),
            "left_shoulder": np.array([0.0, 2.0]),
            "right_shoulder": np.array([2.0, 2.0]),
            "left_hip": np.array([0.0, 5.0]),
            "right_hip": np.array([2.0, 5.0]),
            "left_knee": np.array([0.0, 8.0]),
            "right_knee": np.array([2.0, 8.0]),
            "left_ankle": np.array([0.0, 11.0]),
            "right_ankle": np.array([2.0, 11.0]),
            "left_heel": np.array([0.0, 13.0]),
            "right_heel": np.array([2.0, 13.0]),
        }

        result = HeightCalculator().calculate(keypoints, px_per_cm=2.0)

        self.assertAlmostEqual(result["height_cm"], 6.5)
        self.assertAlmostEqual(result["segments"]["head"], 1.0)
        self.assertAlmostEqual(result["segments"]["torso"], 1.5)
        self.assertAlmostEqual(result["segments"]["leg"], 4.0)
        self.assertEqual(result["segments"]["foot"], 0.0)


if __name__ == "__main__":
    unittest.main()
