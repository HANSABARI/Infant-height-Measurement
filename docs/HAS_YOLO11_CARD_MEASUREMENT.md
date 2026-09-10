# HaS Image Model FP32 Card Segmentation Measurement

## Purpose

The active card reference pipeline uses `xuanwulab/HaS_Image_0209_FP32`, officially titled `HaS Image Model (FP32)`. It is a YOLO11-based Ultralytics instance segmentation model published on Hugging Face under the MIT license.

The model returns bounding boxes and pixel-level masks for sensitive visual regions. H-ALIGN filters the output to these card/document classes:

| Class | ID |
|---|---:|
| `id_card` | 3 |
| `passport` | 5 |
| `employee_badge` | 6 |
| `bank_card` | 8 |

The HaS model can identify all four classes above, but the height-measurement
pipeline accepts only `id_card` and `bank_card` as scale references. Both use
the ISO/IEC 7810 ID-1 long side of `8.56 cm`; passports and employee badges
are rejected because their physical dimensions are different or not guaranteed.

## Runtime Flow

1. Load `sensitive_seg_best.pt` with Ultralytics.
2. Run local inference on the uploaded image.
3. Ignore non-card privacy classes such as `face`, `license_plate`, or `paper`.
4. Select the highest-confidence supported card/document instance above the confidence threshold.
5. Convert that instance mask to a binary mask in the original image size.
6. Fit the card boundary from the mask and measure the average long side in pixels.
7. Convert pixels to centimeters with the ISO card width:

```text
px_per_cm = long_side_px / 8.56
```

The detector keeps the same response fields consumed by the height calculator and debug visualizer: `px_per_cm`, `confidence`, `bbox`, `mask`, `measurement_box`, `long_side_segments`, and geometry metadata.

## Weight Configuration

Default local path:

```text
app/models/has_image_0209_fp32/sensitive_seg_best.pt
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `CARD_HAS_MODEL` | default local path above | Use a specific local `.pt` file |
| `CARD_HAS_AUTO_DOWNLOAD` | `1` | Download from Hugging Face when the local file is missing |
| `CARD_HAS_DEVICE` | unset | Optional Ultralytics device, for example `cuda:0` or `cpu` |
| `CARD_HAS_IMGSZ` | `1920` | Ultralytics inference size; higher values help detect small cards in full-body photos |

The `.pt` file is intentionally not committed because model weights are ignored by the repository.

## Code Locations

| File | Responsibility |
|---|---|
| `app/services/card_detector_has.py` | Ultralytics model loading, class filtering, detector response contract |
| `app/services/card_geometry.py` | Mask conversion, robust boundary fitting, long-side pixel measurement |
| `app/routers/measure.py` | Wires the active card detector into the measurement API |
| `tests/test_card_detector_has.py` | Unit coverage for class filtering and mask-based `px_per_cm` output |
