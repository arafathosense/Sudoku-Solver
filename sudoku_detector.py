"""Sudoku grid detection and cell extraction logic."""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import imutils
import numpy as np

from utils import (
    TransformMetadata,
    extract_digit_from_cell,
    four_point_transform,
    split_into_cells,
)


class SudokuDetector:
    def __init__(self, min_contour_ratio: float = 0.03) -> None:
        self.min_contour_ratio = min_contour_ratio

    def preprocess(self, gray: np.ndarray) -> Dict[str, np.ndarray]:
        equalized = cv2.equalizeHist(gray)
        blurred = cv2.bilateralFilter(equalized, d=7, sigmaColor=75, sigmaSpace=75)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
        thresh = cv2.bitwise_not(thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph = cv2.medianBlur(morph, 5)
        edges = cv2.Canny(morph, 25, 110)
        edges = cv2.dilate(edges, kernel, iterations=1)
        return {"gray": gray, "thresh": morph, "edges": edges}

    def locate_puzzle(
        self,
        frame: np.ndarray,
        previous_transform: Optional[TransformMetadata] = None,
        assume_full_frame: bool = False,
    ) -> Dict[str, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if previous_transform is not None:
            cached = self._warp_with_previous(gray, previous_transform)
            if cached is not None:
                return {
                    "warped": cached,
                    "transform": previous_transform,
                    "contour": previous_transform.rect,
                }

        layers = self.preprocess(gray)
        frame_area = frame.shape[0] * frame.shape[1]
        height, width = frame.shape[:2]
        fallback_rect: Optional[np.ndarray] = None
        for source in ("edges", "thresh"):
            contours = cv2.findContours(layers[source].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < frame_area * self.min_contour_ratio:
                    break
                x, y, w, h = cv2.boundingRect(contour)
                if x <= 5 or y <= 5 or x + w >= width - 5 or y + h >= height - 5:
                    continue
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
                if fallback_rect is None:
                    fallback_rect = cv2.boxPoints(cv2.minAreaRect(contour)).astype("float32")
                if len(approx) == 4:
                    warped, transform = four_point_transform(layers["gray"], approx.reshape(4, 2))
                    if not self._is_squareish(transform.size):
                        continue
                    if not self._has_grid_structure(warped):
                        continue
                    return {
                        "warped": warped,
                        "transform": transform,
                        "contour": approx.reshape(4, 2),
                    }
        if fallback_rect is not None:
            warped, transform = four_point_transform(layers["gray"], fallback_rect)
            if self._is_squareish(transform.size):
                return {
                    "warped": warped,
                    "transform": transform,
                    "contour": fallback_rect,
                }
        if assume_full_frame:
            fallback = self._use_entire_frame(gray)
            if fallback is not None:
                return fallback
        raise ValueError("Sudoku grid not detected")

    def split_cells(self, warped: np.ndarray) -> List[List[np.ndarray]]:
        return split_into_cells(warped)

    def extract_digit(self, cell: np.ndarray):  # -> Optional[np.ndarray]
        return extract_digit_from_cell(cell)

    def _warp_with_previous(self, gray: np.ndarray, transform: TransformMetadata) -> Optional[np.ndarray]:
        try:
            warped = cv2.warpPerspective(gray, transform.matrix, (transform.size[1], transform.size[0]))
        except cv2.error:
            return None
        if self._is_warp_valid(warped):
            return warped
        return None

    def _is_warp_valid(self, warped: np.ndarray) -> bool:
        edges = cv2.Canny(warped, 25, 110)
        edge_pixels = cv2.countNonZero(edges)
        return edge_pixels > warped.size * 0.01

    def _is_squareish(self, size: tuple[int, int]) -> bool:
        height, width = size
        if width == 0:
            return False
        ratio = height / float(width)
        return 0.85 <= ratio <= 1.15

    def _use_entire_frame(self, gray: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        height, width = gray.shape[:2]
        if min(height, width) == 0:
            return None
        rect = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype="float32",
        )
        warped, transform = four_point_transform(gray, rect)
        return {
            "warped": warped,
            "transform": transform,
            "contour": rect,
        }

    def _has_grid_structure(self, warped: np.ndarray) -> bool:
        blur = cv2.GaussianBlur(warped, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
        thresh = cv2.bitwise_not(thresh)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, warped.shape[0] // 30)))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, warped.shape[1] // 30), 1))
        vertical_lines = cv2.erode(thresh, vertical_kernel, iterations=1)
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)
        horizontal_lines = cv2.erode(thresh, horizontal_kernel, iterations=1)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)
        grid = cv2.bitwise_and(vertical_lines, horizontal_lines)
        coverage = cv2.countNonZero(grid) / float(grid.size)
        return coverage > 0.03
