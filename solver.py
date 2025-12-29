"""Backtracking Sudoku solver."""
from __future__ import annotations

from typing import List, Optional, Sequence

Grid = List[List[int]]


class SudokuSolver:
    """Recursive solver with safeguards for invalid or noisy grids."""

    def __init__(self, max_backtracks: Optional[int] = 200_000) -> None:
        self.max_backtracks = max_backtracks
        self._attempts = 0
        self._cutoff = False
        self.last_status: str = "idle"

    def _reset_state(self) -> None:
        self._attempts = 0
        self._cutoff = False
        self.last_status = "idle"

    def solve_board(self, board: Sequence[Sequence[int]]) -> Optional[Grid]:
        self._reset_state()
        working: Grid = [list(row) for row in board]
        if not self._is_consistent(working):
            self.last_status = "invalid"
            return None
        if self._solve(working):
            self.last_status = "solved"
            return working
        self.last_status = "timeout" if self._cutoff else "unsolved"
        return None

    def _find_empty(self, board: Grid) -> Optional[tuple[int, int]]:
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    return row, col
        return None

    def _is_consistent(self, board: Grid) -> bool:
        for row in range(9):
            seen: set[int] = set()
            for value in board[row]:
                if value == 0:
                    continue
                if value in seen:
                    return False
                seen.add(value)

        for col in range(9):
            seen = set()
            for row in range(9):
                value = board[row][col]
                if value == 0:
                    continue
                if value in seen:
                    return False
                seen.add(value)

        for start_row in range(0, 9, 3):
            for start_col in range(0, 9, 3):
                seen = set()
                for r in range(start_row, start_row + 3):
                    for c in range(start_col, start_col + 3):
                        value = board[r][c]
                        if value == 0:
                            continue
                        if value in seen:
                            return False
                        seen.add(value)
        return True

    def _is_valid(self, board: Grid, row: int, col: int, value: int) -> bool:
        if any(board[row][c] == value for c in range(9)):
            return False
        if any(board[r][col] == value for r in range(9)):
            return False
        start_row = (row // 3) * 3
        start_col = (col // 3) * 3
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if board[r][c] == value:
                    return False
        return True

    def _solve(self, board: Grid) -> bool:
        empty = self._find_empty(board)
        if not empty:
            return True
        row, col = empty
        for value in range(1, 10):
            if self.max_backtracks is not None and self._attempts >= self.max_backtracks:
                self._cutoff = True
                return False
            if self._is_valid(board, row, col, value):
                board[row][col] = value
                self._attempts += 1
                if self._solve(board):
                    return True
                board[row][col] = 0
        return False
