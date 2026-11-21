
"""
DeepGuard image prediction adapter.

Provides a framework-independent interface for the
FastAPI backend and other consumers.

No API routes or upload storage logic are implemented here.
"""

import os
import time
from functools import lru_cache
from pathlib import Path

from deepguard_inference import DeepGuardImageDetector


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "deepguard_resnet50_best.pth"
)


@lru_cache(maxsize=1)
def get_detector():
    """
    Load the trained model once per process.
    """

    checkpoint_path = Path(
        os.getenv(
            "DEEPGUARD_MODEL_PATH",
            str(DEFAULT_MODEL_PATH)
        )
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    return DeepGuardImageDetector(
        checkpoint_path=checkpoint_path,
        device="cpu"
    )


def predict_image(image_path: str | Path) -> dict:
    """
    Predict whether an image is REAL or FAKE.

    Args:
        image_path: Path to a valid image file.

    Returns:
        prediction: REAL or FAKE
        confidence: 0-100 model confidence
        probabilities: REAL/FAKE scores on a 0-1 scale
        processing_time: seconds

    This classifier does not provide face counts or
    individual forensic indicator scores.
    """

    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    detector = get_detector()

    start = time.perf_counter()

    # Read image bytes; detector processes in memory.
    image_bytes = path.read_bytes()
    result = detector.predict(image_bytes)

    elapsed = time.perf_counter() - start

    return {
        "prediction": result["prediction"],
        "confidence": round(
            result["confidence"] * 100,
            2
        ),
        "probabilities": result["probabilities"],
        "processing_time": round(elapsed, 3)
    }
