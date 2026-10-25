"""
Video processing service — Phase 1 MOCK implementation.

This module defines the exact interface that the real Sanjeev video-processing
integration must conform to. During Phase 1 every result is randomly generated
demo data so the rest of the system can be developed and tested end-to-end.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTEGRATION NOTE — Phase 3 replacement
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When Sanjeev's video-processing module is ready, replace the body of
process_video() with the real call, for example:

    from video_processing.pipeline import process_video as _real_process

    def process_video(video_path: str | Path) -> dict:
        raw = _real_process(str(video_path))
        return {
            "prediction":        raw["prediction"],           # "REAL" | "FAKE" | "INCONCLUSIVE"
            "confidence":        float(raw["confidence"]),    # 0.0 – 100.0
            "faces_detected":    int(raw.get("faces_detected", 1)),
            "frames_analyzed":   int(raw["frames_analyzed"]),
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


class VideoServiceError(Exception):
    """Raised when the video processing pipeline cannot complete."""


def process_video(video_path: str | Path) -> dict:
    """
    Process a video file and return deepfake detection results.

    Args:
        video_path: Absolute path to the saved video file on disk.

    Returns:
        A dict with the following keys (snake_case — mapped to camelCase by
        response_builder.py):

        prediction          str   "REAL" | "FAKE" | "INCONCLUSIVE"
        confidence          float 0.0 – 100.0
        faces_detected      int   Number of unique faces across sampled frames
        frames_analyzed     int   Number of frames sampled during analysis
        artifacts_detected  int   Number of manipulation artefacts detected
        processing_time     float Wall-clock seconds taken for the full pipeline
        indicators          dict  Five forensic consistency scores (0–100):
                                  face_consistency, texture_consistency,
                                  lighting_consistency, compression_artifacts,
                                  edge_artifacts
        note                str   Human-readable summary sentence
        demo_mode           bool  True while the mock is active

    Raises:
        VideoServiceError: If the video cannot be processed.
    """
    # ------------------------------------------------------------------ #
    #  ⚠️  MOCK IMPLEMENTATION — Phase 1 only.                          #
    #     Replace with real ML inference when Sanjeev's module is        #
    #     integrated (Phase 3).  See module docstring for instructions.  #
    # ------------------------------------------------------------------ #
    _start = time.perf_counter()
    rng = random.Random()   # unseeded → different result every call

    roll = rng.random()
    if roll < 0.42:
        prediction = "FAKE"
        confidence_base = 91.0
        note = (
            "[DEMO — No ML model active] DeepGuard identified temporal inconsistencies "
            "across sampled frames, which are characteristic of synthetic media. "
            "Multiple high-confidence manipulation indicators were detected."
        )
    elif roll < 0.76:
        prediction = "REAL"
        confidence_base = 88.0
        note = (
            "[DEMO — No ML model active] Frame-level analysis found no significant "
            "manipulation indicators. The video appears temporally consistent and "
            "consistent with an authentic recording."
        )
    else:
        prediction = "INCONCLUSIVE"
        confidence_base = 74.0
        note = (
            "[DEMO — No ML model active] Frame-level analysis produced an inconclusive "
            "result. Partial temporal anomalies were detected but did not reach "
            "the threshold for a definitive classification."
        )

    confidence = round(rng.uniform(confidence_base, confidence_base + 5.0), 1)
    faces_detected = max(1, rng.randint(1, 4))
    frames_analyzed = rng.choice([30, 60, 90, 120])
    artifacts_detected = max(3, rng.randint(4, 12))
    elapsed = time.perf_counter() - _start + rng.uniform(1.5, 4.5)

    return {
        "prediction":         prediction,
        "confidence":         confidence,
        "faces_detected":     faces_detected,
        "frames_analyzed":    frames_analyzed,
        "artifacts_detected": artifacts_detected,
        "processing_time":    round(elapsed, 3),
        "indicators": {
            "face_consistency":      rng.randint(68, 97),
            "texture_consistency":   rng.randint(70, 95),
            "lighting_consistency":  rng.randint(63, 91),
            "compression_artifacts": rng.randint(66, 92),
            "edge_artifacts":        rng.randint(74, 97),
        },
        "note":      note,
        "demo_mode": True,
    }
