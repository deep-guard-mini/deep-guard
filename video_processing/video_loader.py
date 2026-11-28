from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class InvalidVideoError(ValueError):
    """Raised when a video file cannot be opened or contains no readable frames."""


def extract_face_frames_with_metadata(
    video_path: str | Path,
    num_frames: int = 32,
    out_size: int = 256,
    pad_ratio: float = 0.25,
) -> tuple[list[np.ndarray], int]:
    """
    Sample evenly spaced frames from a video and extract face crops.

    Falls back to a center square crop when no frontal face is detected in a frame
    so FakeSTormer preprocessing remains consistent across videos without clear faces.

    Returns:
        (crops, faces_detected) where `crops` is a list of `num_frames` RGB arrays
        of shape `(out_size, out_size, 3)` and `faces_detected` is the count of
        sampled frames in which at least one face was detected.
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise InvalidVideoError(f"Could not open video file: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise InvalidVideoError(f"Could not read frames from video: {path}")

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    # Use CascadeClassifier if available in this cv2 build, otherwise fall back cleanly
    face_cascade = None
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    crops: list[np.ndarray] = []
    last_valid_crop: np.ndarray | None = None
    faces_detected = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            continue

        h, w = frame_bgr.shape[:2]
        faces = []
        if face_cascade is not None and not face_cascade.empty():
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

        if len(faces) > 0:
            faces_detected += 1
            x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            pad_w, pad_h = int(fw * pad_ratio), int(fh * pad_ratio)
            x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
            x2, y2 = min(w, x + fw + pad_w), min(h, y + fh + pad_h)
            crop_bgr = frame_bgr[y1:y2, x1:x2]
        else:
            side = min(h, w)
            y1, x1 = (h - side) // 2, (w - side) // 2
            crop_bgr = frame_bgr[y1 : y1 + side, x1 : x1 + side]

        crop_rgb = cv2.cvtColor(
            cv2.resize(crop_bgr, (out_size, out_size)), cv2.COLOR_BGR2RGB
        )
        crops.append(crop_rgb)
        last_valid_crop = crop_rgb

    cap.release()

    if not crops or last_valid_crop is None:
        raise InvalidVideoError(
            f"No valid frames could be extracted from the video: {path}"
        )

    while len(crops) < num_frames:
        crops.append(last_valid_crop.copy())

    return crops[:num_frames], faces_detected


def extract_face_frames(
    video_path: str | Path,
    num_frames: int = 32,
    out_size: int = 256,
    pad_ratio: float = 0.25,
) -> list[np.ndarray]:
    """Extract `num_frames` RGB face/center crops from a video file."""
    crops, _ = extract_face_frames_with_metadata(
        video_path=video_path,
        num_frames=num_frames,
        out_size=out_size,
        pad_ratio=pad_ratio,
    )
    return crops
