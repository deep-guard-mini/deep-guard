"""
Image processing service — Phase 1 MOCK implementation.

This module defines the exact interface that the real Pavankumar image-processing
integration must conform to. During Phase 1 every result is randomly generated
demo data so the rest of the system can be developed and tested end-to-end.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTEGRATION NOTE — Phase 3 replacement
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When Pavankumar's image-processing module is ready, replace the body of
process_image() with the real call, for example:

    from image_processing.predictor import predict_image as _real_predict

    def process_image(image_path: str | Path) -> dict:
        raw = _real_predict(str(image_path))
        return {
            "prediction":        raw["prediction"],        # "REAL" | "FAKE" | "INCONCLUSIVE"
            "confidence":        float(raw["confidence"]), # 0.0 – 100.0
            "faces_detected":    int(raw.get("faces_analyzed", 1)),
            "frames_analyzed":   1,
            "artifacts_detected": int(raw.get("artifacts_detected", 0)),
            "processing_time":   float(raw.get("processing_time", 0.0)),
            "indicators": {
                "face_consistency":      int(raw["indicators"]["face_consistency"]),
                "texture_consistency":   int(raw["indicators"]["texture_consistency"]),
                "lighting_consistency":  int(raw["indicators"]["lighting_consistency"]),
                "compression_artifacts": int(raw["indicators"]["compression_artifacts"]),
                "edge_artifacts":        int(raw["indicators"]["edge_artifacts"]),
            },
            "note":      raw.get("note", ""),
            "demo_mode": False,   # ← set to False once real inference is active
        }

The returned dict MUST contain all keys listed in the docstring below.
Do NOT change the key names — response_builder.py maps them to the frontend contract.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import random
import time
from pathlib import Path


class ImageServiceError(Exception):
    """Raised when the image processing pipeline cannot complete."""


def process_image(image_path: str | Path) -> dict:
    """
    Process an image file and return deepfake detection results.

    Args:
        image_path: Absolute path to the saved image file on disk.

    Returns:
        A dict with the following keys (snake_case — mapped to camelCase by
        response_builder.py):

        prediction          str   "REAL" | "FAKE" | "INCONCLUSIVE"
        confidence          float 0.0 – 100.0
        faces_detected      int   Number of faces identified in the image
        frames_analyzed     int   Always 1 for still images
        artifacts_detected  int   Number of manipulation artefacts found
        processing_time     float Wall-clock seconds taken for inference
        indicators          dict  Five forensic consistency scores (0–100):
                                  face_consistency, texture_consistency,
                                  lighting_consistency, compression_artifacts,
                                  edge_artifacts
        note                str   Human-readable summary sentence
        demo_mode           bool  True while the mock is active

    Raises:
        ImageServiceError: If the image cannot be processed.
    """
    # ------------------------------------------------------------------ #
    #  ⚠️  MOCK IMPLEMENTATION — Phase 1 only.                          #
    #     Replace with real ML inference when Pavankumar's module is     #
    #     integrated (Phase 3).  See module docstring for instructions.  #
    # ------------------------------------------------------------------ #
    _start = time.perf_counter()
    rng = random.Random()   # unseeded → different result every call

    roll = rng.random()
    if roll < 0.42:
        prediction = "FAKE"
        confidence_base = 94.0
        note = (
            "[DEMO — No ML model active] DeepGuard identified visual characteristics "
            "associated with potential media manipulation. Several analysed indicators "
            "showed elevated anomaly scores."
        )
    elif roll < 0.76:
        prediction = "REAL"
        confidence_base = 90.0
        note = (
            "[DEMO — No ML model active] No significant manipulation indicators "
            "were detected. The analysed media appears consistent with an authentic source."
        )
    else:
        prediction = "INCONCLUSIVE"
        confidence_base = 76.0
        note = (
            "[DEMO — No ML model active] Analysis returned an inconclusive result. "
            "Some indicators are elevated but do not meet the threshold for a "
            "definitive classification."
        )

    confidence = round(rng.uniform(confidence_base, confidence_base + 5.0), 1)
    faces_detected = max(1, rng.randint(1, 3))
    artifacts_detected = max(3, rng.randint(4, 10))
    elapsed = time.perf_counter() - _start + rng.uniform(0.3, 1.1)

    return {
        "prediction":         prediction,
        "confidence":         confidence,
        "faces_detected":     faces_detected,
        "frames_analyzed":    1,
        "artifacts_detected": artifacts_detected,
        "processing_time":    round(elapsed, 3),
        "indicators": {
            "face_consistency":      rng.randint(70, 98),
            "texture_consistency":   rng.randint(72, 96),
            "lighting_consistency":  rng.randint(65, 92),
            "compression_artifacts": rng.randint(68, 93),
            "edge_artifacts":        rng.randint(76, 98),
        },
        "note":      note,
        "demo_mode": True,
    }
