"""Instant Sudoku solver driven by a live webcam feed."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import imutils
import numpy as np

from digit_recognizer import DigitRecognizer
from solver import SudokuSolver
from sudoku_detector import SudokuDetector
from utils import TransformMetadata, project_overlay, render_detected_digits, render_solution_overlay


MIN_CONFIDENCE = 0.75
MIN_DIGITS_TO_SOLVE = 17
HOLD_SECONDS = 5.0
FRAME_WIDTH = 820
REACQUIRE_INTERVAL = 5
BoardSignature = Tuple[Tuple[int, ...], ...]


def process_frame(
    frame: np.ndarray,
    detector: SudokuDetector,
    recognizer: DigitRecognizer,
    solver: SudokuSolver,
    previous_transform: Optional[TransformMetadata],
    allow_cached_transform: bool,
    previous_board_signature: Optional[BoardSignature],
    assume_full_frame: bool,
    cell_dump_dir: Optional[Path] = None,
    dump_cells: bool = False,
) -> tuple[np.ndarray, Dict[str, Optional[object]]]:
    display = frame.copy()
    message = "Searching for Sudoku grid..."
    status_color = (0, 165, 255)
    recognized_cells = 0
    info: Dict[str, Optional[object]] = {
        "transform": previous_transform,
        "solved": False,
        "overlay": None,
        "message": message,
        "status_color": status_color,
        "recognized_cells": recognized_cells,
        "board_signature": None,
        "attempted_solve": False,
        "warped": None,
        "dumped_cells": False,
    }

    try:
        puzzle = detector.locate_puzzle(
            frame,
            previous_transform if allow_cached_transform else None,
            assume_full_frame=assume_full_frame,
        )
        warped = puzzle["warped"]
        transform = puzzle["transform"]
        info["transform"] = transform
        info["warped"] = warped

        cells = detector.split_cells(warped)
        dump_raw_dir: Optional[Path] = None
        dump_digits_dir: Optional[Path] = None
        if cell_dump_dir is not None and dump_cells:
            dump_raw_dir = cell_dump_dir / "raw"
            dump_digits_dir = cell_dump_dir / "digits"
            dump_raw_dir.mkdir(parents=True, exist_ok=True)
            dump_digits_dir.mkdir(parents=True, exist_ok=True)

        detected_board: list[list[int]] = []
        given_mask: list[list[bool]] = [[False] * 9 for _ in range(9)]
        digit_requests: list[tuple[int, int, np.ndarray]] = []

        for row_idx, row_cells in enumerate(cells):
            detected_row: list[int] = []
            for col_idx, cell in enumerate(row_cells):
                if dump_raw_dir is not None:
                    cv2.imwrite(str(dump_raw_dir / f"{row_idx}_{col_idx}.png"), cell)
                digit_img = detector.extract_digit(cell)
                if digit_img is None:
                    detected_row.append(0)
                    continue
                if dump_digits_dir is not None:
                    cv2.imwrite(str(dump_digits_dir / f"{row_idx}_{col_idx}.png"), digit_img)
                digit_requests.append((row_idx, col_idx, digit_img))
                detected_row.append(0)
            detected_board.append(detected_row)

        if dump_raw_dir is not None:
            info["dumped_cells"] = True

        if digit_requests:
            predictions = recognizer.predict_batch([req[2] for req in digit_requests], MIN_CONFIDENCE)
            for (row_idx, col_idx, _), (digit, confidence) in zip(digit_requests, predictions):
                if confidence >= MIN_CONFIDENCE and digit != 0:
                    detected_board[row_idx][col_idx] = digit
                    given_mask[row_idx][col_idx] = True
                    recognized_cells += 1

        if recognized_cells:
            detected_overlay = render_detected_digits(warped.shape[:2], detected_board)
            display = project_overlay(display, detected_overlay, transform.inverse)

        board_signature: Optional[BoardSignature] = tuple(tuple(row) for row in detected_board) if recognized_cells else None
        info["board_signature"] = board_signature
        info["recognized_cells"] = recognized_cells

        solved = None
        board_cached = bool(board_signature and board_signature == previous_board_signature)
        should_attempt = bool(board_signature) and recognized_cells >= MIN_DIGITS_TO_SOLVE and not board_cached
        if should_attempt:
            solved = solver.solve_board(detected_board)
        info["attempted_solve"] = should_attempt

        if solved:
            overlay = render_solution_overlay(warped.shape[:2], solved, detected_board, given_mask)
            display = project_overlay(display, overlay, transform.inverse)
            info["solved"] = True
            info["overlay"] = overlay
            message = "Sudoku solved – hold steady"
            status_color = (0, 255, 0)
        elif board_cached and recognized_cells >= MIN_DIGITS_TO_SOLVE:
            message = "Solution cached – keep puzzle steady"
            status_color = (0, 255, 0)
        else:
            status_color = (0, 0, 255)
            if recognized_cells == 0:
                message = "Hold the full Sudoku in frame"
            elif recognized_cells < MIN_DIGITS_TO_SOLVE:
                message = f"Need clearer view ({recognized_cells}/81 digits read)"
            else:
                message = "Grid detected but solver needs a cleaner scan"
    except ValueError:
        info["transform"] = None
        info["board_signature"] = None
        info["recognized_cells"] = 0
        info["attempted_solve"] = False
        info["warped"] = None
        message = "Searching for Sudoku grid..."
        status_color = (0, 165, 255)

    info["message"] = message
    info["status_color"] = status_color
    cv2.putText(display, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)
    return display, info


def run(
    camera_index: int,
    image_path: Optional[str] = None,
    assume_full_frame: bool = False,
    dump_cells_dir: Optional[str] = None,
) -> None:
    detector = SudokuDetector()
    recognizer = DigitRecognizer()
    solver = SudokuSolver()

    image_mode = image_path is not None
    raw_image: Optional[np.ndarray] = None
    cap: Optional[cv2.VideoCapture] = None
    if image_mode:
        raw_image = cv2.imread(image_path)  # type: ignore[arg-type]
        if raw_image is None:
            raise FileNotFoundError(f"Unable to read image at {image_path}")
    else:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open camera index {camera_index}")

    cached_transform: Optional[TransformMetadata] = None
    hold_overlay: Optional[np.ndarray] = None
    hold_transform: Optional[TransformMetadata] = None
    hold_until = 0.0
    frame_idx = 0
    last_board_signature: Optional[BoardSignature] = None
    cell_dump_path: Optional[Path] = None
    cells_dumped = False
    if dump_cells_dir:
        cell_dump_path = Path(dump_cells_dir).expanduser().resolve()
        if cell_dump_path.exists():
            shutil.rmtree(cell_dump_path)
        cell_dump_path.mkdir(parents=True, exist_ok=True)

    cv2.namedWindow("Instant Sudoku Solver", cv2.WINDOW_NORMAL)

    while True:
        if image_mode:
            if raw_image is None:
                break
            frame = raw_image.copy()
            grabbed = True
        else:
            assert cap is not None
            grabbed, frame = cap.read()
            if not grabbed:
                break
        frame_idx += 1
        frame = imutils.resize(frame, width=FRAME_WIDTH)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        allow_cached = not image_mode and cached_transform is not None and frame_idx % REACQUIRE_INTERVAL != 0

        output_frame, info = process_frame(
            frame,
            detector,
            recognizer,
            solver,
            cached_transform,
            allow_cached,
            last_board_signature,
            assume_full_frame or image_mode,
            cell_dump_path,
            dump_cells=bool(cell_dump_path and not cells_dumped),
        )

        transform = info.get("transform")
        if isinstance(transform, TransformMetadata):
            cached_transform = transform
        else:
            cached_transform = None

        board_sig = info.get("board_signature")
        if board_sig is not None:
            last_board_signature = board_sig  # type: ignore[assignment]

        if info.get("solved") and isinstance(info.get("overlay"), np.ndarray) and isinstance(transform, TransformMetadata):
            hold_overlay = info["overlay"]  # type: ignore[assignment]
            hold_transform = transform
            hold_until = time.time() + HOLD_SECONDS
        elif hold_overlay is not None and hold_transform is not None and time.time() < hold_until:
            output_frame = project_overlay(output_frame, hold_overlay, hold_transform.inverse)
            cv2.putText(
                output_frame,
                "Solution locked – move on when ready",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        elif hold_overlay is not None and time.time() >= hold_until:
            hold_overlay = None
            hold_transform = None

        if info.get("dumped_cells"):
            cells_dumped = True

        cv2.imshow("Sudoku Solver", output_frame)
        if image_mode:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instant Sudoku Solver using OpenCV and TensorFlow")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--image", type=str, default=None, help="Path to a static Sudoku image (overrides webcam)")
    parser.add_argument(
        "--assume-full-frame",
        action="store_true",
        help="Treat the entire frame as the puzzle if detection fails",
    )
    parser.add_argument(
        "--dump-cells",
        type=str,
        default=None,
        help="Directory to save warped cell crops for debugging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.camera,
        args.image,
        assume_full_frame=args.assume_full_frame,
        dump_cells_dir=args.dump_cells,
    )
