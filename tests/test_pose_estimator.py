import unittest

import numpy as np

from app.services.pose_estimator import PoseEstimator


class PoseEstimatorTest(unittest.TestCase):
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
