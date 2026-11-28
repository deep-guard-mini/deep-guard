"""Tests for FakeSTormer video-processing module and backend/image regressions."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image
from torchvision.models import resnet50

from backend.app import app
from video_processing.fakestormer_inference import FakeSTormerVideoDetector
from video_processing.predictor import get_video_detector, predict_video, predict_video_file
from video_processing.video_loader import (
    InvalidVideoError,
    extract_face_frames,
    extract_face_frames_with_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_CANDIDATES = [
    REPO_ROOT / "models" / "fakestormer_c23_best.pth",
    REPO_ROOT
    / "models"
    / "TopDownDetector_C23_ViTBase224_ST_hm100_tempLOC0.2_4SBI_SAM_mp0.01_temp2_0vidAug_0.35_temp_normlms_normREAL_model_best.pth",
]


def _find_checkpoint() -> Path | None:
    for p in CHECKPOINT_CANDIDATES:
        if p.is_file():
            return p
    return None


def _write_synthetic_video(path: Path, num_frames: int = 12, size: int = 128) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (size, size)
    )
    for i in range(num_frames):
        frame = np.full((size, size, 3), fill_value=(i * 15) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_missing_checkpoint_raises_clear_error(tmp_path: Path):
    missing_ckpt = tmp_path / "missing_checkpoint.pth"
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        FakeSTormerVideoDetector(checkpoint_path=missing_ckpt)


def test_missing_config_raises_clear_error(tmp_path: Path):
    dummy_ckpt = tmp_path / "dummy.pth"
    dummy_ckpt.write_bytes(b"placeholder")
    missing_cfg = tmp_path / "missing_config.yaml"
    with pytest.raises(FileNotFoundError, match="configuration file not found"):
        FakeSTormerVideoDetector(checkpoint_path=dummy_ckpt, config_path=missing_cfg)


def test_invalid_and_missing_video_raise_errors(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        extract_face_frames(tmp_path / "does_not_exist.mp4")

    corrupt_video = tmp_path / "corrupt.mp4"
    corrupt_video.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 128)
    with pytest.raises(InvalidVideoError):
        extract_face_frames(corrupt_video)


def test_video_without_faces_falls_back_to_center_crop(tmp_path: Path):
    no_face_video = _write_synthetic_video(tmp_path / "no_face.mp4", num_frames=12)
    frames, faces_detected = extract_face_frames_with_metadata(
        no_face_video, num_frames=16, out_size=256
    )
    assert len(frames) == 16
    assert faces_detected == 0
    for frame in frames:
        assert frame.shape == (256, 256, 3)
        assert frame.dtype == np.uint8


def test_real_fakestormer_model_inference_and_backend(tmp_path: Path):
    ckpt = _find_checkpoint()
    if ckpt is None:
        pytest.skip("FakeSTormer C23 checkpoint not present on disk")

    get_video_detector.cache_clear()
    detector = FakeSTormerVideoDetector(checkpoint_path=ckpt, device="cpu")
    assert detector.cfg.MODEL.type == "TopDownDetector"
    assert detector.num_frames == 4

    # Verify preprocessing transforms produce expected normalized (C, H, W) tensor
    sample_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    tensor = detector.transform(sample_rgb)
    assert tensor.shape == (3, 224, 224)

    # Inference on synthetic video without faces (CPU)
    no_face_video = _write_synthetic_video(tmp_path / "no_face.mp4", num_frames=12)
    res_no_face = predict_video(no_face_video)
    assert res_no_face["prediction"] in {"REAL", "FAKE"}
    assert 0.0 <= res_no_face["real_probability"] <= 1.0
    assert 0.0 <= res_no_face["fake_probability"] <= 1.0
    assert abs((res_no_face["real_probability"] + res_no_face["fake_probability"]) - 1.0) < 1e-3
    assert res_no_face["frames_analyzed"] == 16
    assert res_no_face["faces_detected"] == 0

    # Compatibility alias matches
    res_alias = predict_video_file(no_face_video)
    assert res_alias["prediction"] == res_no_face["prediction"]
    assert res_alias["frames_analyzed"] == 16

    # FastAPI /api/detect/video integration on CPU
    client = TestClient(app)
    video_bytes = no_face_video.read_bytes()
    resp = client.post(
        "/api/detect/video",
        files={"file": ("no_face.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["mediaType"] == "video"
    assert body["prediction"] in {"REAL", "FAKE"}
    assert body["framesAnalyzed"] == 16
    assert body["demoMode"] is False
    assert set(body["probabilities"].keys()) == {"REAL", "FAKE"}
    assert abs((body["probabilities"]["REAL"] + body["probabilities"]["FAKE"]) - 1.0) < 1e-3


def test_image_processing_pipeline_still_works(tmp_path: Path):
    """Verify existing image_processing + /api/detect/image contract remains intact."""
    import image_processing.predictor as img_predictor
    from image_processing.deepguard_inference import DeepGuardImageDetector

    # Build a temporary ResNet50 checkpoint matching DeepGuardImageDetector's schema
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    ckpt_path = tmp_path / "resnet50_test.pth"
    torch.save(
        {
            "class_mapping": {"REAL": 0, "FAKE": 1},
            "image_size": 224,
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )

    detector = DeepGuardImageDetector(checkpoint_path=ckpt_path, device="cpu")
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color="green").save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    client = TestClient(app)
    with patch.object(img_predictor, "get_detector", return_value=detector):
        resp = client.post(
            "/api/detect/image",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["mediaType"] == "image"
    assert data["prediction"] in {"REAL", "FAKE"}
    assert data["framesAnalyzed"] == 1
    assert data["demoMode"] is False
    assert abs((data["probabilities"]["REAL"] + data["probabilities"]["FAKE"]) - 1.0) < 1e-3
