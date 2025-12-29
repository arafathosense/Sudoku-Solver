"""Utility helpers for perspective transforms, cell extraction, and overlays."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from matplotlib import cm

# Precompute a soft-green palette (leveraging matplotlib dependency)
_DIGIT_COLORS = (cm.get_cmap("Greens")(np.linspace(0.35, 0.95, 10))[:, :3] * 255).astype("uint8")


@dataclass
class TransformMetadata:
    matrix: np.ndarray
    inverse: np.ndarray
    rect: np.ndarray
    dst: np.ndarray
    size: Tuple[int, int]


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points in top-left, top-right, bottom-right, bottom-left order."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, TransformMetadata]:
    """Warp the supplied image using a four-point perspective transform."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    inverse = cv2.getPerspectiveTransform(dst, rect)
    return warped, TransformMetadata(matrix=matrix, inverse=inverse, rect=rect, dst=dst, size=(max_height, max_width))


def split_into_cells(warped: np.ndarray) -> List[List[np.ndarray]]:
    """Divide the warped puzzle into 81 cell images."""
    grid: List[List[np.ndarray]] = []
    h, w = warped.shape[:2]
    cell_h = h // 9
    cell_w = w // 9

    for y in range(9):
        row = []
        for x in range(9):
            start_y = y * cell_h
            end_y = (y + 1) * cell_h
            start_x = x * cell_w
            end_x = (x + 1) * cell_w
            row.append(warped[start_y:end_y, start_x:end_x])
        grid.append(row)
    return grid


def _prepare_cell(cell: np.ndarray) -> np.ndarray:
    """Apply thresholding tuned for well-exposed webcam feeds."""
    cell = cv2.GaussianBlur(cell, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        cell,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    thresh = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    margin = int(min(thresh.shape[:2]) * 0.06)
    if margin > 0:
        thresh = thresh[margin:-margin or None, margin:-margin or None]
    return thresh


def _prepare_cell_fallback(cell: np.ndarray) -> np.ndarray:
    """Alternative cleanup tuned for glossy phone screens."""
    cell = cv2.GaussianBlur(cell, (5, 5), 0)
    _, thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    margin = int(min(cleaned.shape[:2]) * 0.05)
    if margin > 0:
        cleaned = cleaned[margin:-margin or None, margin:-margin or None]
    return cleaned


def _prepare_cell_print(cell: np.ndarray) -> np.ndarray:
    """Cleanup specialized for high-contrast printed puzzles."""
    blurred = cv2.GaussianBlur(cell, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    thick = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    return thick


def _touches_border(x: int, y: int, w: int, h: int, width: int, height: int) -> bool:
    margin = max(int(min(width, height) * 0.08), 1)
    left = x <= margin
    top = y <= margin
    right = x + w >= width - margin
    bottom = y + h >= height - margin
    touches = sum(int(flag) for flag in (left, top, right, bottom))
    return touches >= 2


def _normalize_digit_roi(roi: np.ndarray) -> Optional[np.ndarray]:
    """Pad and scale the digit ROI to MNIST-like 28x28 framing."""
    if roi.size == 0:
        return None
    coords = cv2.findNonZero(roi)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    roi = roi[y : y + h, x : x + w]
    if roi.size == 0:
        return None
    height, width = roi.shape[:2]
    if height == 0 or width == 0:
        return None
    scale = 20.0 / max(height, width)
    resized = cv2.resize(
        roi,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((28, 28), dtype="uint8")
    y_offset = (28 - resized.shape[0]) // 2
    x_offset = (28 - resized.shape[1]) // 2
    canvas[y_offset : y_offset + resized.shape[0], x_offset : x_offset + resized.shape[1]] = resized
    return canvas


def _extract_from_binary(processed: np.ndarray, min_area_ratio: float) -> Optional[np.ndarray]:
    height, width = processed.shape[:2]
    min_pixels = max(int(min_area_ratio * (height * width)), 1)
    contours, _ = cv2.findContours(processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_pixels:
                break
            x, y, w, h = cv2.boundingRect(contour)
            if _touches_border(x, y, w, h, width, height) and max(w, h) < max(width, height) * 0.8:
                continue
            mask = np.zeros(processed.shape, dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)
            digit = cv2.bitwise_and(processed, processed, mask=mask)
            roi = digit[y : y + h, x : x + w]
            normalized = _normalize_digit_roi(roi)
            if normalized is not None:
                return normalized

    coords = cv2.findNonZero(processed)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    if w * h < max(min_pixels // 2, 1):
        return None
    if _touches_border(x, y, w, h, width, height) and max(w, h) < max(width, height) * 0.8:
        return None
    roi = processed[y : y + h, x : x + w]
    return _normalize_digit_roi(roi)


def extract_digit_from_cell(cell: np.ndarray) -> Optional[np.ndarray]:
    """Return a 28x28 digit image if a number is present, otherwise None."""
    pipelines = (
        (_prepare_cell, 0.02),
        (_prepare_cell_fallback, 0.02),
        (_prepare_cell_print, 0.025),
    )
    for generator, min_ratio in pipelines:
        processed = generator(cell)
        digit = _extract_from_binary(processed, min_ratio)
        if digit is not None:
            return digit
    return None


def render_solution_overlay(
    shape: Tuple[int, int],
    solved: Sequence[Sequence[int]],
    detected: Sequence[Sequence[int]],
    given_mask: Optional[Sequence[Sequence[bool]]] = None,
) -> np.ndarray:
    """Render solved digits (that were blank) into a colored overlay image."""
    height, width = shape
    overlay = np.zeros((height, width, 3), dtype="uint8")
    cell_h = height / 9
    cell_w = width / 9

    for row in range(9):
        for col in range(9):
            value = solved[row][col]
            if value == 0:
                continue
            if given_mask and given_mask[row][col]:
                continue
            if detected[row][col] != 0:
                continue
            color = tuple(int(channel) for channel in _DIGIT_COLORS[value])
            text = str(value)
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            text_x = int(col * cell_w + (cell_w - text_size[0]) / 2)
            text_y = int(row * cell_h + (cell_h + text_size[1]) / 2)
            cv2.putText(
                overlay,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA,
            )
    return overlay


def render_detected_digits(shape: Tuple[int, int], detected: Sequence[Sequence[int]]) -> np.ndarray:
    """Draw recognized digits so the user can see what the model thinks is present."""
    height, width = shape
    overlay = np.zeros((height, width, 3), dtype="uint8")
    cell_h = height / 9
    cell_w = width / 9
    for row in range(9):
        for col in range(9):
            value = detected[row][col]
            if value == 0:
                continue
            text = str(value)
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            text_x = int(col * cell_w + (cell_w - text_size[0]) / 2)
            text_y = int(row * cell_h + (cell_h + text_size[1]) / 2)
            cv2.putText(
                overlay,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
    return overlay


def project_overlay(frame: np.ndarray, overlay: np.ndarray, inverse_matrix: np.ndarray) -> np.ndarray:
    """Project the overlay back into the original camera frame."""
    warped = cv2.warpPerspective(overlay, inverse_matrix, (frame.shape[1], frame.shape[0]))
    mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(frame, frame, mask=mask_inv)
    combined = cv2.add(background, warped)
    return combined
