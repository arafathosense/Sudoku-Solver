# Sudoku Solver

Instant Sudoku Solver is a personal computer-vision playground from tubakhxn that proves a lean laptop webcam plus a lightweight TensorFlow model can perform end-to-end puzzle solving in real time. It combines traditional image processing, a compact CNN trained on MNIST, and a classic depth-first search solver to deliver a quick demo-worthy experience without depending on any cloud APIs.

## How it works
1. **Webcam capture** – Frames are resized, converted to grayscale, and binarized with adaptive thresholding.
2. **Grid detection** – Canny edges plus contour analysis locate the largest square; a four-point perspective transform yields a frontal view of the puzzle.
3. **Cell processing** – The warped grid is split into 81 cells. Each cell undergoes cleanup, contour filtering, and digit isolation.
4. **Digit recognition** – Cropped digits are passed to a TensorFlow CNN trained on MNIST (model auto-trains once, then loads from `models/mnist_cnn.keras`). Empty cells are treated as zeros.
5. **Sudoku solving** – A classic depth-first backtracking solver validates rows, columns, and 3×3 boxes until the grid is complete.
6. **Augmented overlay** – Newly solved digits are rendered in green using a Matplotlib-derived palette, inverse-warped, and blended back with the live feed.

## Project layout
```
main.py               # Entry point with webcam loop
sudoku_detector.py    # Grid detection and warping helpers
digit_recognizer.py   # CNN loading, optional training, digit prediction
solver.py             # Backtracking Sudoku solver
utils.py              # Perspective math, cell extraction, overlay utilities
requirements.txt      # Python dependencies
README.md             # Documentation
models/               # Auto-created folder for the saved CNN weights
```

## Setup
1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure a webcam is available and unobstructed.

> **Note:** The first run trains the CNN on MNIST (≈1–2 minutes depending on hardware). Subsequent runs load the saved model instantly.

## Usage
```bash
python main.py --camera 0
```
- Hold a Sudoku grid (printed, on paper, or on a phone screen) flat within the frame.
- Wait until the overlay turns green with "Sudoku solved".
- Press `q` to exit.

## Forking & contributions
1. Fork this repository into your GitHub account so you can freely experiment without affecting the source.
2. Create a feature branch before making changes (`git checkout -b feature/my-idea`).
3. Keep pull requests focused and include before/after screenshots if you tweak the UI overlay.
4. Credit tubakhxn when you publish derivatives so others can discover the original project.

## Limitations
- Needs good lighting and minimal glare; reflective phone screens may require angling the display.
- Assumes standard 9×9 Sudoku puzzles with clear cell boundaries.
- Digit recognizer is MNIST-based and may misread stylized fonts.
- Training step requires internet access the first time (to download MNIST).

## Future improvements
- Add temporal smoothing to stabilize predictions across frames.
- Support manual corrections via keyboard/mouse before solving.
- Bundle a pre-trained lightweight ONNX model to skip the initial training pass.
- Provide a fallback OCR path when the CNN confidence is low.

## 👤 Author

**HOSEN ARAFAT**  

**Software Engineer, China**  

**GitHub:** https://github.com/arafathosense

**Researcher: Artificial Intelligence, Machine Learning, Deep Learning, Computer Vision, Image Processing**

