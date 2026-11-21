"""DeepGuard video inference pipeline.

Input: a local video file path.
Output: video-level REAL / FAKE / INCONCLUSIVE prediction and metadata.

The pipeline mirrors the inference procedure used to evaluate the trained
DeepGuard ResNet-50 checkpoint:
- sample up to 10 frames evenly across the video;
- detect faces with OpenCV Haar Cascade;
- resize face crops to 224x224 and apply ImageNet normalization;
- infer with ResNet-50 (class order: real=0, fake=1);
- average fake-class probabilities across all detected face crops;
- classify as FAKE when the mean fake probability is >= 0.5.

The model checkpoint is not included in this source package. Place it at
models/deepguard_multimanip_best.pth in the repository root, or set the
DEEPGUARD_MODEL_PATH environment variable.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


class VideoProcessingError(RuntimeError):
    """Raised when a video cannot be processed by the DeepGuard pipeline."""


_MODEL: torch.nn.Module | None = None
_DEVICE: torch.device | None = None
_MODEL_PATH: Path | None = None
_MODEL_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()

_IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def _resolve_model_path() -> Path:
    """Resolve checkpoint from env var or the repository's models folder."""
    configured = os.getenv("DEEPGUARD_MODEL_PATH")
    if configured:
        return Path(configured).expanduser().resolve()

    # video_processing/pipeline.py -> repository root -> models/checkpoint
    repository_root = Path(__file__).resolve().parents[1]
    return repository_root / "models" / "deepguard_multimanip_best.pth"


def _load_model() -> tuple[torch.nn.Module, torch.device]:
    """Load the trained ResNet-50 once per process (lazy, cached loading)."""
    global _MODEL, _DEVICE, _MODEL_PATH

    with _MODEL_LOCK:
        requested_path = _resolve_model_path()
        if _MODEL is not None and _DEVICE is not None and _MODEL_PATH == requested_path:
            return _MODEL, _DEVICE

        if not requested_path.is_file():
            raise VideoProcessingError(
                "DeepGuard checkpoint not found. Expected "
                f"'{requested_path}'. Copy deepguard_multimanip_best.pth into "
                "the repository models/ folder or set DEEPGUARD_MODEL_PATH."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # The project checkpoint contains tensors and basic Python values.
            checkpoint = torch.load(
                requested_path,
                map_location="cpu",
                weights_only=True,
            )
            class_names = checkpoint.get("class_names", ["real", "fake"])
            if list(class_names) != ["real", "fake"]:
                raise VideoProcessingError(
                    "Unexpected checkpoint class order. Expected ['real', 'fake'], "
                    f"received {class_names!r}."
                )

            network = models.resnet50(weights=None)
            network.fc = torch.nn.Linear(2048, 2)
            network.load_state_dict(checkpoint["model_state_dict"], strict=True)
            network.to(device)
            network.eval()
        except VideoProcessingError:
            raise
        except Exception as exc:  # noqa: BLE001 - convert model errors to module error
            raise VideoProcessingError(
                f"Could not load DeepGuard checkpoint '{requested_path}': {exc}"
            ) from exc

        _MODEL = network
        _DEVICE = device
        _MODEL_PATH = requested_path
        return _MODEL, _DEVICE


def _get_face_cascade() -> cv2.CascadeClassifier:
    cascade_path = (
        Path(cv2.data.haarcascades)
        / "haarcascade_frontalface_default.xml"
    )
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise VideoProcessingError(
            f"Could not load OpenCV Haar Cascade from '{cascade_path}'."
        )
    return cascade


def _sample_frame_indices(total_frames: int, max_frames: int) -> np.ndarray:
    if total_frames <= 0:
        raise VideoProcessingError("The video contains no readable frames.")
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero.")
    return np.linspace(
        0,
        total_frames - 1,
        num=min(max_frames, total_frames),
        dtype=int,
    )


def _predict_face_batch(
    face_tensors: list[torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> list[float]:
    """Return fake-class probabilities for all face crops."""
    if not face_tensors:
        return []

    probabilities: list[float] = []
    with _INFERENCE_LOCK, torch.inference_mode():
        for start in range(0, len(face_tensors), batch_size):
            batch = torch.stack(face_tensors[start : start + batch_size]).to(
                device,
                non_blocking=(device.type == "cuda"),
            )

            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(batch)
            else:
                logits = model(batch)

            fake_probs = torch.softmax(logits.float(), dim=1)[:, 1]
            probabilities.extend(fake_probs.detach().cpu().tolist())

    return probabilities


def process_video(
    video_path: str | Path,
    *,
    max_frames: int = 10,
    threshold: float = 0.5,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Analyze one video and return the backend video-service contract.

    Args:
        video_path: Path to a video already saved on local disk.
        max_frames: Maximum number of evenly spaced frames to sample.
        threshold: Fixed fake-probability decision threshold (default 0.5).
        batch_size: Number of face crops per model inference batch.

    Returns:
        A dictionary compatible with backend/services/video_service.py:
        prediction, confidence, faces_detected, frames_analyzed,
        artifacts_detected, processing_time, indicators, note, demo_mode.
        Extra fake_probability and real_probability fields are also included.

    Notes:
        - confidence is the predicted class's mean probability, expressed as
          a percentage (0-100), not a calibrated probability of correctness.
        - faces_detected counts detected face crops across sampled frames; it
          does not track unique people across time.
        - forensic indicator scores and artifacts_detected are not computed by
          this ResNet-50 classifier. They are returned as zero only to satisfy
          the current backend response schema; do not present them as measured
          forensic evidence.
        - if no faces are detected, prediction is INCONCLUSIVE.

    Raises:
        VideoProcessingError: If the file, OpenCV decoder, face detector, or
            model cannot be used.
        ValueError: If an invalid threshold or batch/frame setting is supplied.
    """
    started = time.perf_counter()
    path = Path(video_path).expanduser().resolve()

    if not path.is_file():
        raise VideoProcessingError(f"Video file does not exist: '{path}'.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    model, device = _load_model()
    cascade = _get_face_cascade()

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoProcessingError(f"OpenCV could not open video: '{path}'.")

    face_tensors: list[torch.Tensor] = []
    frames_analyzed = 0

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = _sample_frame_indices(total_frames, max_frames)

        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = capture.read()
            if not success or frame is None:
                continue

            frames_analyzed += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )

            height, width = frame.shape[:2]
            for x, y, w, h in detected:
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2 = min(width, int(x + w))
                y2 = min(height, int(y + h))
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(face_rgb)
                face_tensors.append(_IMAGE_TRANSFORM(pil_image))
    except VideoProcessingError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise VideoProcessingError(f"Video frame processing failed: {exc}") from exc
    finally:
        capture.release()

    faces_detected = len(face_tensors)
    elapsed = round(time.perf_counter() - started, 3)

    if frames_analyzed == 0:
        raise VideoProcessingError(
            "No video frames could be decoded. Check the video file/codec."
        )

    # The model is not needed for an INCONCLUSIVE no-face result, but it was
    # loaded above so a missing checkpoint is reported consistently.
    if faces_detected == 0:
        return {
            "prediction": "INCONCLUSIVE",
            "confidence": 0.0,
            "fake_probability": None,
            "real_probability": None,
            "faces_detected": 0,
            "frames_analyzed": frames_analyzed,
            "artifacts_detected": 0,
            "processing_time": elapsed,
            "indicators": {
                "face_consistency": 0,
                "texture_consistency": 0,
                "lighting_consistency": 0,
                "compression_artifacts": 0,
                "edge_artifacts": 0,
            },
            "note": (
                "No faces were detected in the sampled frames. DeepGuard could "
                "not produce a reliable face-based prediction. Forensic indicator "
                "scores are not computed by this classifier."
            ),
            "demo_mode": False,
        }

    fake_probabilities = _predict_face_batch(
        face_tensors,
        model,
        device,
        batch_size,
    )
    mean_fake_probability = float(np.mean(fake_probabilities))
    mean_real_probability = 1.0 - mean_fake_probability

    prediction = "FAKE" if mean_fake_probability >= threshold else "REAL"
    predicted_class_probability = (
        mean_fake_probability
        if prediction == "FAKE"
        else mean_real_probability
    )

    elapsed = round(time.perf_counter() - started, 3)
    return {
        "prediction": prediction,
        "confidence": round(predicted_class_probability * 100.0, 2),
        "fake_probability": round(mean_fake_probability * 100.0, 2),
        "real_probability": round(mean_real_probability * 100.0, 2),
        "faces_detected": faces_detected,
        "frames_analyzed": frames_analyzed,
        "artifacts_detected": 0,
        "processing_time": elapsed,
        "indicators": {
            "face_consistency": 0,
            "texture_consistency": 0,
            "lighting_consistency": 0,
            "compression_artifacts": 0,
            "edge_artifacts": 0,
        },
        "note": (
            f"DeepGuard sampled {frames_analyzed} frames and analyzed "
            f"{faces_detected} detected face crops. Mean fake-class score: "
            f"{mean_fake_probability * 100.0:.2f}%; decision threshold: "
            f"{threshold * 100.0:.0f}%. The score is not calibrated certainty. "
            "Forensic indicator scores are not computed by this classifier."
        ),
        "demo_mode": False,
    }
