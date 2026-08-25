import unittest

import numpy as np

from app.services.card_detector_has import (
    CARD_CLASS_IDS,
    CARD_REFERENCE_WIDTH_CM,
    HaSCardDetector,
)


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeBoxes:
    def __init__(self, classes, confidences, boxes):
        self.cls = FakeTensor(classes)
        self.conf = FakeTensor(confidences)
        self.xyxy = FakeTensor(boxes)


class FakeMasks:
    def __init__(self, masks):
        self.data = FakeTensor(masks)


class FakeResult:
    names = {
        0: "face",
        CARD_CLASS_IDS["id_card"]: "id_card",
        CARD_CLASS_IDS["bank_card"]: "bank_card",
    }

    def __init__(self, boxes, masks):
        self.boxes = boxes
        self.masks = masks


class FakeModel:
    names = FakeResult.names

    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, source, conf, imgsz, verbose):
        self.calls.append(
            {"source": source, "conf": conf, "imgsz": imgsz, "verbose": verbose}
        )
        return [self.result]


class FakeSequenceModel:
    names = FakeResult.names

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def predict(self, source, conf, imgsz, verbose):
        self.calls.append(
            {"source": source, "conf": conf, "imgsz": imgsz, "verbose": verbose}
        )
        return [self.results.pop(0)]


class HaSCardDetectorTests(unittest.TestCase):
    def test_detect_filters_to_supported_card_classes_and_uses_mask_geometry_for_scale(self):
        face_mask = np.zeros((10, 10), dtype=np.uint8)
        id_card_mask = np.zeros((10, 10), dtype=np.uint8)
        id_card_mask[:5, :5] = 1
        bank_card_mask = np.zeros((10, 10), dtype=np.uint8)
        bank_card_mask[5:, 5:] = 1
        result = FakeResult(
            boxes=FakeBoxes(
                classes=[0, CARD_CLASS_IDS["id_card"], CARD_CLASS_IDS["bank_card"]],
                confidences=[0.99, 0.2, 0.88],
                boxes=[
                    [0.0, 0.0, 9.0, 9.0],
                    [1.0, 1.0, 7.0, 7.0],
                    [2.0, 2.0, 8.0, 8.0],
                ],
            ),
            masks=FakeMasks([face_mask, id_card_mask, bank_card_mask]),
        )
        model = FakeModel(result)
        geometry_calls = []

        def geometry_fn(mask):
            geometry_calls.append(mask.copy())
            return {
                "box": np.asarray(
                    [[2.0, 2.0], [173.2, 2.0], [173.2, 56.0], [2.0, 56.0]],
                    dtype=np.float32,
                ),
                "long_side_points": np.asarray(
                    [[2.0, 2.0], [173.2, 2.0]],
                    dtype=np.float32,
                ),
                "long_side_segments": np.asarray(
                    [
                        [[2.0, 2.0], [173.2, 2.0]],
                        [[2.0, 56.0], [173.2, 56.0]],
                    ],
                    dtype=np.float32,
                ),
                "long_side_lengths_px": np.asarray([171.2, 171.2], dtype=np.float32),
                "long_side_px": 171.2,
                "method": "test_geometry",
            }

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        detector = HaSCardDetector(
            model=model,
            model_path="unused.pt",
            score_threshold=0.25,
            geometry_fn=geometry_fn,
        )

        detected = detector.detect(img)

        self.assertTrue(detected["detected"])
        self.assertEqual(detected["class_name"], "bank_card")
        self.assertEqual(detected["class_id"], CARD_CLASS_IDS["bank_card"])
        self.assertEqual(detected["bbox"], [2.0, 2.0, 8.0, 8.0])
        self.assertEqual(detected["confidence"], 0.88)
        self.assertEqual(detected["px_per_cm"], 171.2 / CARD_REFERENCE_WIDTH_CM)
        self.assertEqual(detected["reference_length_cm"], CARD_REFERENCE_WIDTH_CM)
        self.assertEqual(detected["geometry_method"], "test_geometry")
        self.assertTrue(detected["used_mask"])
        np.testing.assert_array_equal(geometry_calls[0], bank_card_mask)
        self.assertEqual(model.calls[0]["conf"], 0.25)
        self.assertEqual(model.calls[0]["imgsz"], 1920)
        self.assertFalse(model.calls[0]["verbose"])
        self.assertIs(model.calls[0]["source"], img)

    def test_detect_returns_not_detected_when_only_non_card_classes_are_present(self):
        result = FakeResult(
            boxes=FakeBoxes(
                classes=[0],
                confidences=[0.99],
                boxes=[[0.0, 0.0, 9.0, 9.0]],
            ),
            masks=FakeMasks([np.ones((10, 10), dtype=np.uint8)]),
        )
        detector = HaSCardDetector(
            model=FakeModel(result),
            model_path="unused.pt",
            score_threshold=0.25,
        )

        detected = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

        self.assertFalse(detected["detected"])
        self.assertEqual(detected["px_per_cm"], 0.0)
        self.assertEqual(detected["confidence"], 0.0)

    def test_detect_rejects_passport_for_credit_card_scale_measurement(self):
        result = FakeResult(
            boxes=FakeBoxes(
                classes=[CARD_CLASS_IDS["passport"]],
                confidences=[0.99],
                boxes=[[0.0, 0.0, 9.0, 9.0]],
            ),
            masks=FakeMasks([np.ones((10, 10), dtype=np.uint8)]),
        )
        detector = HaSCardDetector(
            model=FakeModel(result),
            model_path="unused.pt",
            score_threshold=0.25,
        )

        detected = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))

        self.assertFalse(detected["detected"])
        self.assertEqual(detected["px_per_cm"], 0.0)

    def test_detect_uses_tile_fallback_and_maps_card_result_to_original_image(self):
        full_result = FakeResult(
            boxes=FakeBoxes(
                classes=[0],
                confidences=[0.99],
                boxes=[[0.0, 0.0, 40.0, 40.0]],
            ),
            masks=FakeMasks([np.ones((80, 80), dtype=np.uint8)]),
        )
        tile_mask = np.zeros((60, 50), dtype=np.uint8)
        tile_mask[20:40, 10:30] = 1
        tile_result = FakeResult(
            boxes=FakeBoxes(
                classes=[CARD_CLASS_IDS["bank_card"]],
                confidences=[0.79],
                boxes=[[10.0, 20.0, 30.0, 40.0]],
            ),
            masks=FakeMasks([tile_mask]),
        )
        model = FakeSequenceModel([full_result, tile_result])
        geometry_masks = []

        def geometry_fn(mask):
            geometry_masks.append(mask.copy())
            return {
                "box": np.asarray(
                    [[110.0, 220.0], [281.2, 220.0], [281.2, 274.0], [110.0, 274.0]],
                    dtype=np.float32,
                ),
                "long_side_points": np.asarray(
                    [[110.0, 220.0], [281.2, 220.0]],
                    dtype=np.float32,
                ),
                "long_side_segments": np.asarray(
                    [
                        [[110.0, 220.0], [281.2, 220.0]],
                        [[110.0, 274.0], [281.2, 274.0]],
                    ],
                    dtype=np.float32,
                ),
                "long_side_lengths_px": np.asarray([171.2, 171.2], dtype=np.float32),
                "long_side_px": 171.2,
                "method": "tile_geometry",
            }

        detector = HaSCardDetector(
            model=model,
            model_path="unused.pt",
            score_threshold=0.25,
            tile_regions=[(100, 200, 150, 260)],
            geometry_fn=geometry_fn,
        )

        detected = detector.detect(np.zeros((300, 400, 3), dtype=np.uint8))

        self.assertTrue(detected["detected"])
        self.assertEqual(detected["bbox"], [110.0, 220.0, 130.0, 240.0])
        self.assertEqual(detected["detection_source"], "tile")
        self.assertEqual(detected["tile_region"], [100, 200, 150, 260])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[1]["source"].shape, (60, 50, 3))
        self.assertEqual(geometry_masks[0].shape, (300, 400))
        np.testing.assert_array_equal(
            geometry_masks[0][200:260, 100:150],
            tile_mask,
        )
        self.assertEqual(int(geometry_masks[0].sum()), int(tile_mask.sum()))


if __name__ == "__main__":
    unittest.main()
