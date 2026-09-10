from __future__ import annotations

import numpy as np


def mask_to_numpy(mask, image_shape: tuple[int, int]) -> np.ndarray:
    if hasattr(mask, "cpu"):
        mask = mask.cpu()
    if hasattr(mask, "numpy"):
        mask = mask.numpy()
    elif hasattr(mask, "to_ndarray"):
        mask = mask.to_ndarray()
    else:
        mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = mask[0]

    mask = (mask > 0.5).astype(np.uint8)
    if mask.shape[:2] != image_shape:
        import cv2

        mask = cv2.resize(
            mask,
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return mask


def compute_mask_geometry(mask: np.ndarray | None) -> dict | None:
    if mask is None:
        return None

    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 10:
        return None

    rect = cv2.minAreaRect(contour)
    (_, _), (width, height), _ = rect
    if width <= 0 or height <= 0:
        return None

    initial_box = cv2.boxPoints(rect).astype(np.float32)
    contour_points = contour.reshape(-1, 2).astype(np.float32)
    fitted_lines = []

    for edge_idx in range(4):
        start = initial_box[edge_idx]
        end = initial_box[(edge_idx + 1) % 4]
        edge = end - start
        edge_length_sq = float(np.dot(edge, edge))
        if edge_length_sq <= 0:
            return _min_area_geometry(initial_box)

        relative = contour_points - start
        projection = (relative @ edge) / edge_length_sq
        cross = np.abs(edge[0] * relative[:, 1] - edge[1] * relative[:, 0])
        distance = cross / np.sqrt(edge_length_sq)

        side_points = contour_points[
            (projection >= 0.12)
            & (projection <= 0.88)
            & (distance <= max(3.0, 0.08 * np.sqrt(edge_length_sq)))
        ]
        if len(side_points) < 6:
            return _min_area_geometry(initial_box)

        fitted = cv2.fitLine(
            side_points.reshape(-1, 1, 2),
            cv2.DIST_HUBER,
            0,
            0.01,
            0.01,
        ).reshape(-1)
        fitted_lines.append(
            (
                np.array([fitted[2], fitted[3]], dtype=np.float32),
                np.array([fitted[0], fitted[1]], dtype=np.float32),
            )
        )

    corners = []
    for edge_idx in range(4):
        previous_line = fitted_lines[(edge_idx - 1) % 4]
        current_line = fitted_lines[edge_idx]
        corner = _line_intersection(previous_line, current_line)
        if corner is None:
            return _min_area_geometry(initial_box)
        corners.append(corner)

    box = np.asarray(corners, dtype=np.float32)
    contour_area = float(cv2.contourArea(contour))
    fitted_area = abs(float(cv2.contourArea(box)))
    if (
        not np.isfinite(box).all()
        or not cv2.isContourConvex(box.astype(np.int32))
        or fitted_area < contour_area * 0.65
        or fitted_area > contour_area * 1.35
    ):
        return _min_area_geometry(initial_box)

    return _geometry_from_box(box, method="robust_line_fit")


def _line_intersection(
    first_line: tuple[np.ndarray, np.ndarray],
    second_line: tuple[np.ndarray, np.ndarray],
) -> np.ndarray | None:
    first_point, first_direction = first_line
    second_point, second_direction = second_line
    denominator = (
        first_direction[0] * second_direction[1]
        - first_direction[1] * second_direction[0]
    )
    if abs(float(denominator)) < 1e-6:
        return None

    delta = second_point - first_point
    distance = (
        delta[0] * second_direction[1]
        - delta[1] * second_direction[0]
    ) / denominator
    return first_point + distance * first_direction


def _min_area_geometry(box: np.ndarray) -> dict:
    return _geometry_from_box(
        box.astype(np.float32),
        method="min_area_rect_fallback",
    )


def _geometry_from_box(box: np.ndarray, method: str) -> dict:
    edges = np.roll(box, -1, axis=0) - box
    edge_lengths = np.linalg.norm(edges, axis=1)
    first_pair_mean = float((edge_lengths[0] + edge_lengths[2]) / 2)
    second_pair_mean = float((edge_lengths[1] + edge_lengths[3]) / 2)
    long_edge_indices = (0, 2) if first_pair_mean >= second_pair_mean else (1, 3)

    long_side_segments = np.asarray(
        [
            [box[index], box[(index + 1) % 4]]
            for index in long_edge_indices
        ],
        dtype=np.float32,
    )
    long_side_lengths = edge_lengths[list(long_edge_indices)].astype(np.float32)
    long_side_px = float(np.mean(long_side_lengths))

    return {
        "box": box.astype(np.float32),
        "long_side_points": long_side_segments[0],
        "long_side_segments": long_side_segments,
        "long_side_lengths_px": long_side_lengths,
        "long_side_px": long_side_px,
        "method": method,
    }
