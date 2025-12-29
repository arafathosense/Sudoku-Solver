"""Digit recognizer powered by a lightweight CNN trained on MNIST."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import tensorflow as tf

_TEMPLATE_FONT_CONFIGS = (
    (cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2),
    (cv2.FONT_HERSHEY_SIMPLEX, 0.85, 1),
    (cv2.FONT_HERSHEY_DUPLEX, 0.85, 2),
    (cv2.FONT_HERSHEY_COMPLEX, 0.8, 2),
)
_TEMPLATE_THRESHOLD = 0.8
_TEMPLATE_MARGIN = 0.12


class DigitRecognizer:
    def __init__(self, model_path: str = "models/mnist_cnn.keras") -> None:
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = self._load_or_train_model()
        self.templates = self._build_templates()

    def _build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.MaxPool2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPool2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def _load_or_train_model(self) -> tf.keras.Model:
        if self.model_path.exists():
            return tf.keras.models.load_model(self.model_path)
        model = self._build_model()
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
        x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
        model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=3,
            batch_size=128,
            verbose=1,
        )
        model.save(self.model_path)
        return model

    def _build_templates(self) -> dict[int, list[np.ndarray]]:
        templates: dict[int, list[np.ndarray]] = {}
        for digit in range(1, 10):
            variants: list[np.ndarray] = []
            text = str(digit)
            for font, scale, thickness in _TEMPLATE_FONT_CONFIGS:
                canvas = np.zeros((28, 28), dtype="uint8")
                text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
                text_w, text_h = text_size
                x = max((28 - text_w) // 2, 0)
                y = max((28 + text_h) // 2, baseline)
                cv2.putText(canvas, text, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
                variants.append(canvas.astype("float32") / 255.0)
            templates[digit] = variants
        return templates

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_vec = a.flatten()
        b_vec = b.flatten()
        denom = (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)) + 1e-8
        if denom == 0.0:
            return 0.0
        return float(np.dot(a_vec, b_vec) / denom)

    def _template_predict(self, digit_img: np.ndarray) -> Tuple[int, float]:
        normalized = (digit_img.astype("float32") / 255.0)
        best_digit = 0
        best_score = 0.0
        for digit, variants in self.templates.items():
            for template in variants:
                score = self._cosine_similarity(normalized, template)
                if score > best_score:
                    best_digit = digit
                    best_score = score
        return best_digit, best_score

    def _predict_probabilities(self, batch: np.ndarray) -> np.ndarray:
        return self.model.predict(batch, verbose=0)

    def predict_batch(self, digits: Sequence[np.ndarray], min_confidence: float = 0.6) -> List[Tuple[int, float]]:
        if not digits:
            return []
        batch = np.stack([(img.astype("float32") / 255.0)[..., np.newaxis] for img in digits])
        probabilities = self._predict_probabilities(batch)
        results: List[Tuple[int, float]] = []
        for img, probs in zip(digits, probabilities):
            digit = int(np.argmax(probs))
            confidence = float(probs[digit])
            template_digit, template_score = self._template_predict(img)
            template_ok = template_score >= _TEMPLATE_THRESHOLD

            if digit == 0 or confidence < min_confidence:
                if template_ok:
                    digit = template_digit
                    confidence = template_score
                else:
                    digit = 0
            elif template_ok and (template_score - confidence) >= _TEMPLATE_MARGIN:
                digit = template_digit
                confidence = template_score

            results.append((digit if digit != 0 else 0, confidence))
        return results

    def predict_digit(self, digit_img: np.ndarray, min_confidence: float = 0.6) -> Tuple[int, float]:
        batch_result = self.predict_batch([digit_img], min_confidence=min_confidence)
        return batch_result[0] if batch_result else (0, 0.0)


def _load_digit_sample(path: str) -> np.ndarray:
    digit = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if digit is None:
        raise FileNotFoundError(f"Unable to load sample at {path}")
    digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
    return digit


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Inspect digit crops with the MNIST CNN")
    parser.add_argument("--samples", nargs="+", help="Paths to cropped digit images to evaluate")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum confidence threshold (default: 0.0)")
    args = parser.parse_args()

    if not args.samples:
        parser.error("Provide at least one image path via --samples")

    recognizer = DigitRecognizer()
    digits: list[np.ndarray] = []
    paths: list[str] = []
    for sample_path in args.samples:
        try:
            digits.append(_load_digit_sample(sample_path))
            paths.append(sample_path)
        except FileNotFoundError as exc:
            print(exc)

    if not digits:
        raise SystemExit("No valid digit samples to evaluate.")

    predictions = recognizer.predict_batch(digits, min_confidence=args.min_confidence)
    for path, (digit, confidence) in zip(paths, predictions):
        print(f"{path}: predicted {digit} (confidence {confidence:.2f})")


if __name__ == "__main__":
    _cli()
