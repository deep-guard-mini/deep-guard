"""
DeepGuard video processing service.

Connects the FastAPI backend to the video inference pipeline
(video_processing.pipeline), adapting raw pipeline outputs into
the backend service response contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_processing.pipeline import VideoProcessingError
from video_processing.pipeline import process_video as _real_process


class VideoServiceError(Exception):
    """Raised when the video processing pipeline cannot complete."""


def process_video(video_path: str | Path) -> dict[str, Any]:
    """
    Process a video file and return deepfake detection results.

    Invokes the video processing pipeline to extract frames, detect faces,
    and infer REAL / FAKE / INCONCLUSIVE predictions using the trained model.

    Args:
        video_path: Path to the saved video file on disk.

    Returns:
        A dict with snake_case keys mapped to the API contract by response_builder:
            prediction          str   "REAL" | "FAKE" | "INCONCLUSIVE"
            confidence          float 0.0 – 100.0
            faces_detected      int | None
            frames_analyzed     int | None
            artifacts_detected  int | None
            processing_time     float Wall-clock seconds
            indicators          dict  Forensic scores (None when not computed)
            probabilities       dict | None {"real": float, "fake": float}
            note                str   Human-readable summary
            demo_mode           bool  False (real ML inference)

    Raises:
        VideoServiceError: If the video cannot be processed.
    """
    try:
        raw = _real_process(str(video_path))
    except VideoProcessingError as exc:
        raise VideoServiceError(f"Video processing failed: {exc}") from exc
    except Exception as exc:
        raise VideoServiceError(f"Unexpected video processing error: {exc}") from exc

    prediction = str(raw.get("prediction", "INCONCLUSIVE"))
    confidence = float(raw.get("confidence", 0.0))
    processing_time = float(raw.get("processing_time", 0.0))
    note = str(raw.get("note", ""))

    faces_detected = (
        int(raw["faces_detected"])
        if raw.get("faces_detected") is not None
        else None
    )
    frames_analyzed = (
        int(raw["frames_analyzed"])
        if raw.get("frames_analyzed") is not None
        else None
    )
    artifacts_detected = (
        int(raw["artifacts_detected"])
        if raw.get("artifacts_detected") is not None
        else None
    )

    real_probability = raw.get("real_probability")
    fake_probability = raw.get("fake_probability")

    probabilities: dict[str, float] | None = None
    if real_probability is not None and fake_probability is not None:
        probabilities = {
            "real": float(real_probability),
            "fake": float(fake_probability),
        }
    elif isinstance(raw.get("probabilities"), dict):
        probabilities = raw["probabilities"]

    return {
        "prediction": prediction,
        "confidence": confidence,
        "faces_detected": faces_detected,
        "frames_analyzed": frames_analyzed,
        "artifacts_detected": artifacts_detected,
        "processing_time": processing_time,
        "indicators": {
            "face_consistency": None,
            "texture_consistency": None,
            "lighting_consistency": None,
            "compression_artifacts": None,
            "edge_artifacts": None,
        },
        "probabilities": probabilities,
        "note": note,
        "demo_mode": False,
    }
