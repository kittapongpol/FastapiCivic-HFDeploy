from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import settings


class InvalidImageError(ValueError):
    """Raised when the uploaded payload cannot be decoded as an image."""


def _load_image(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("Uploaded file is not a valid image.") from exc
    return image


def _fallback_scores(image: Image.Image) -> np.ndarray:
    """Deterministic lightweight classifier used when ONNX weights are absent."""
    resized = image.resize((64, 64))
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    edges = np.abs(np.diff(arr, axis=1)).mean(axis=(0, 1))
    features = np.array(
        [
            means[0],
            means[1],
            means[2],
            stds.mean(),
            edges.mean(),
            arr[:, :32].mean() - arr[:, 32:].mean(),
            arr[:32, :].mean() - arr[32:, :].mean(),
        ],
        dtype=np.float32,
    )
    weights = np.array(
        [
            [1.4, 0.5, 0.2, 0.8, 1.1, -0.4, 0.3],
            [1.1, 0.7, 0.4, 1.0, 1.2, -0.2, 0.2],
            [0.8, 0.9, 0.6, 1.1, 1.0, 0.0, 0.1],
            [0.5, 1.0, 0.9, 1.2, 0.9, 0.2, -0.1],
            [0.4, 0.8, 1.2, 1.0, 0.8, 0.3, -0.2],
            [0.6, 0.6, 1.4, 0.9, 1.1, 0.5, -0.3],
            [0.9, 0.5, 1.1, 0.8, 1.3, 0.7, -0.4],
            [0.7, 0.7, 1.0, 1.1, 1.4, 0.4, -0.2],
        ],
        dtype=np.float32,
    )
    return weights @ features


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def classify_image(image_bytes: bytes, model_path: str | Path | None = None) -> dict:
    image = _load_image(image_bytes)
    resolved_model = Path(model_path or settings.model_path)

    runtime = "fallback-numpy"
    scores = _fallback_scores(image)

    if resolved_model.exists():
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(resolved_model), providers=["CPUExecutionProvider"])
            arr = np.asarray(image.resize((224, 224)), dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))[None, ...]
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: arr})[0][0]
            scores = np.asarray(output, dtype=np.float32)
            runtime = "onnxruntime-quantized"
        except Exception:
            runtime = "fallback-numpy"

    probabilities = _softmax(scores[: len(settings.labels)])
    ranked = sorted(
        zip(settings.labels, probabilities.tolist(), strict=False),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "runtime": runtime,
        "predictions": [{"label": label, "confidence": round(float(conf), 4)} for label, conf in ranked],
    }
