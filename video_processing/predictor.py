from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

import requests

from .fakestormer_inference import FakeSTormerVideoDetector

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "fakestormer_c23_best.pth"
ALT_C23_CHECKPOINT = (
    MODELS_DIR
    / "TopDownDetector_C23_ViTBase224_ST_hm100_tempLOC0.2_4SBI_SAM_mp0.01_temp2_0vidAug_0.35_temp_normlms_normREAL_model_best.pth"
)
DEFAULT_DROPBOX_URL = os.getenv("FAKESTORMER_WEIGHTS_URL", "")


def _resolve_checkpoint_path() -> Path:
    env_path = os.getenv("FAKESTORMER_MODEL_PATH") or os.getenv(
        "DEEPGUARD_VIDEO_MODEL_PATH"
    )
    if env_path:
        return Path(env_path)

    if DEFAULT_MODEL_PATH.is_file():
        return DEFAULT_MODEL_PATH

    if ALT_C23_CHECKPOINT.is_file():
        return ALT_C23_CHECKPOINT

    c23_matches = sorted(MODELS_DIR.glob("TopDownDetector_C23_*_model_best.pth"))
    if c23_matches:
        return c23_matches[0]

    return DEFAULT_MODEL_PATH


def _download_weights_if_missing(dest_path: Path, url: str) -> None:
    if dest_path.is_file() and dest_path.stat().st_size > 10_000_000:
        return
    if dest_path.is_file() and not url:
        return
    if not url:
        raise FileNotFoundError(
            f"FakeSTormer checkpoint not found at '{dest_path}'. "
            "Place the trained C23 weights at 'models/fakestormer_c23_best.pth' "
            "or set FAKESTORMER_MODEL_PATH / FAKESTORMER_WEIGHTS_URL."
        )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(".pth.tmp")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp_path.replace(dest_path)


@lru_cache(maxsize=1)
def get_video_detector() -> FakeSTormerVideoDetector:
    checkpoint_path = _resolve_checkpoint_path()
    weights_url = os.getenv("FAKESTORMER_WEIGHTS_URL", DEFAULT_DROPBOX_URL)
    _download_weights_if_missing(checkpoint_path, weights_url)
    config_path = os.getenv("FAKESTORMER_CONFIG_PATH") or None
    return FakeSTormerVideoDetector(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )


def predict_video(video_path: str | Path) -> dict:
    """
    Run FakeSTormer video-level deepfake detection on a video file.

    Returns:
        prediction: "REAL" or "FAKE"
        label: "REAL" or "FAKE"
        confidence: 0-100 predicted-class percentage
        real_probability: 0-1 score
        fake_probability: 0-1 score
        probabilities: {"REAL": float, "FAKE": float}
        frames_analyzed: number of frames sampled and analyzed
        faces_detected: number of sampled frames with a detected face
        processing_time: wall-clock seconds
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    detector = get_video_detector()
    start = time.perf_counter()
    result = detector.predict_video(path)
    elapsed = time.perf_counter() - start

    real_prob = float(result["real_probability"])
    fake_prob = float(result["fake_probability"])
    label = result["label"]
    confidence = round(max(real_prob, fake_prob) * 100.0, 2)

    return {
        "prediction": label,
        "label": label,
        "confidence": confidence,
        "real_probability": real_prob,
        "fake_probability": fake_prob,
        "probabilities": {
            "REAL": real_prob,
            "FAKE": fake_prob,
        },
        "frames_analyzed": int(result["frames_analyzed"]),
        "faces_detected": int(result["faces_detected"]),
        "processing_time": round(elapsed, 3),
    }


def predict_video_file(video_path: str | Path) -> dict:
    """Compatibility wrapper matching the MINI predictor interface."""
    return predict_video(video_path)
