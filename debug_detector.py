import cv2
from sudoku_detector import SudokuDetector

img = cv2.imread('s 1.png')
if img is None:
    raise SystemExit('image missing')

detector = SudokuDetector()
try:
    puzzle = detector.locate_puzzle(img)
    print('detected', puzzle['warped'].shape)
except Exception as exc:
    print('fail', exc)
