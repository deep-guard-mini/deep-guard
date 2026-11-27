"""
DeepGuard video processing service.

Connects the FastAPI backend to the trained FakeSTormer C23 video
deepfake detection model.

Flow:
    Uploaded video
        -> predict_video()
        -> FakeSTormer (TopDownDetector + TimeViT)
        -> REAL / FAKE prediction + probabilities
        -> Standardized service dict returned to backend
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from video_processing.predictor import predict_video
from video_processing.video_loader import InvalidVideoError


class VideoServiceError(Exception):
    """Raised when the video processing pipeline cannot complete."""


def process_video(video_path: str | Path) -> dict[str, Any]:
    """
    Analyze a video file using DeepGuard's trained FakeSTormer C23 model.

    Invokes the video processing pipeline to extract frames, detect faces,
    and infer REAL / FAKE predictions using the trained model.

    Args:
        video_path: Path to the uploaded video saved temporarily on disk.

    Returns:
        Dict matching `response_builder.build_detection_response()` contract.

    Raises:
        InvalidVideoError: If the uploaded video cannot be decoded.
        VideoServiceError: If model loading or inference fails.
    """
    try:
        raw = predict_video(video_path)
    except InvalidVideoError:
        raise
    except Exception as exc:
        raise VideoServiceError(f"Video inference failed: {exc}") from exc

    prediction = raw["prediction"]
    confidence = float(raw["confidence"])
    real_prob = float(raw["real_probability"])
    fake_prob = float(raw["fake_probability"])
    frames_analyzed = int(raw["frames_analyzed"])
    faces_detected = int(raw.get("faces_detected", 0))
    processing_time = float(raw["processing_time"])

    face_note = (
        f" Detected faces in {faces_detected}/{frames_analyzed} sampled frames."
        if faces_detected > 0
        else " No frontal face was detected; center-crop fallback was used."
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "faces_detected": faces_detected,
        "frames_analyzed": frames_analyzed,
        "artifacts_detected": None,
        "processing_time": processing_time,
        "indicators": {
            "face_consistency": None,
            "texture_consistency": None,
            "lighting_consistency": None,
            "compression_artifacts": None,
            "edge_artifacts": None,
        },
        "probabilities": {
            "REAL": real_prob,
            "FAKE": fake_prob,
        },
        "real_probability": real_prob,
        "fake_probability": fake_prob,
        "note": (
            f"DeepGuard's trained FakeSTormer C23 video model predicted "
            f"{prediction} with {confidence:.2f}% confidence across "
            f"{frames_analyzed} sampled frames.{face_note} This result is an "
            f"AI estimate and does not guarantee video authenticity."
        ),
        "demo_mode": False,
    }
