"""
DeepGuard backend test suite — Phase 1.

Run from the repository root:
    pytest                  # runs all tests
    pytest -v               # verbose
    pytest tests/test_backend.py::TestHealth  # single class

All tests use FastAPI's synchronous TestClient (no real HTTP sockets needed).
Minimal valid "files" are constructed from raw magic-byte headers so no real
media files are required on disk.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from PIL import Image

from backend.app import app
from backend.services.image_service import ImageServiceError
from backend.services.video_service import VideoServiceError

# ---------------------------------------------------------------------------
# Test client
# ---------------------------------------------------------------------------
client = TestClient(app)

# ---------------------------------------------------------------------------
# Valid file payloads
#
# Generates minimal valid images (JPEG, PNG, WebP) using PIL so that both
# file_validation magic-byte detection and real ML model decoding succeed.
# ---------------------------------------------------------------------------
def _generate_test_image(format_name: str, color: str = "blue") -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (32, 32), color=color)
    img.save(buf, format=format_name)
    return buf.getvalue()

JPEG_BYTES = _generate_test_image("JPEG")
PNG_BYTES  = _generate_test_image("PNG")
WEBP_BYTES = _generate_test_image("WEBP")

def _generate_test_video(suffix: str, fourcc: str) -> bytes:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), 10, (64, 64))
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        for _ in range(8):
            writer.write(frame)
        writer.release()
        return Path(path).read_bytes()
    finally:
        if os.path.exists(path):
            os.remove(path)


# Valid magic bytes with corrupted payload (for testing inference error handling)
CORRUPT_JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 128
CORRUPT_MP4_BYTES  = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 128

MP4_BYTES  = _generate_test_video(".mp4", "mp4v")
MOV_BYTES  = _generate_test_video(".mov", "mp4v")
AVI_BYTES  = _generate_test_video(".avi", "MJPG")

# An unsupported format — PDF header
PDF_BYTES  = b"%PDF-1.4\n" + b"\x00" * 128


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------
_REQUIRED_TOP_LEVEL_FIELDS = {
    "status",
    "prediction",
    "confidence",
    "assessment",
    "assessmentTone",
    "mediaType",
    "processingTime",
    "facesDetected",
    "framesAnalyzed",
    "artifactsDetected",
    "indicators",
    "note",
    "demoMode",
}

_REQUIRED_INDICATOR_FIELDS = {
    "faceConsistency",
    "textureConsistency",
    "lightingConsistency",
    "compressionArtifacts",
    "edgeArtifacts",
}

_VALID_PREDICTIONS    = {"REAL", "FAKE", "INCONCLUSIVE"}
_VALID_ASSESSMENTS    = {"LIKELY AUTHENTIC", "LIKELY MANIPULATED", "INCONCLUSIVE"}
_VALID_TONES          = {"safe", "danger", "amber"}


def _assert_valid_detection_response(data: dict, expected_media_type: str) -> None:
    """Assert that a detection response matches the full frontend contract."""
    # ── All top-level fields present ──────────────────────────────────────
    missing_fields = _REQUIRED_TOP_LEVEL_FIELDS - data.keys()
    assert not missing_fields, f"Response missing fields: {missing_fields}"

    # ── status ───────────────────────────────────────────────────────────
    assert data["status"] == "success", f"Expected 'success', got {data['status']!r}"

    # ── prediction ───────────────────────────────────────────────────────
    assert data["prediction"] in _VALID_PREDICTIONS, (
        f"Invalid prediction: {data['prediction']!r}"
    )

    # ── confidence: 0.0–100.0 ────────────────────────────────────────────
    assert isinstance(data["confidence"], (int, float)), "confidence must be numeric"
    assert 0.0 <= data["confidence"] <= 100.0, (
        f"confidence out of range: {data['confidence']}"
    )

    # ── assessment and tone ───────────────────────────────────────────────
    assert data["assessment"] in _VALID_ASSESSMENTS
    assert data["assessmentTone"] in _VALID_TONES

    # ── mediaType ────────────────────────────────────────────────────────
    assert data["mediaType"] == expected_media_type

    # ── processingTime ────────────────────────────────────────────────────
    assert isinstance(data["processingTime"], (int, float)), "processingTime must be numeric"
    assert data["processingTime"] >= 0, f"processingTime must be >= 0, got {data['processingTime']}"

    # ── Optional/mode-specific fields ─────────────────────────────────────
    if expected_media_type == "video":
        assert isinstance(data["framesAnalyzed"], int) and data["framesAnalyzed"] > 0
        assert isinstance(data["facesDetected"], int) and data["facesDetected"] >= 0
        assert data["artifactsDetected"] is None
        assert data["demoMode"] is False, "demoMode should be False for real ML model"

        indicators = data["indicators"]
        assert isinstance(indicators, dict), "indicators must be a dict"
        missing_ind = _REQUIRED_INDICATOR_FIELDS - indicators.keys()
        assert not missing_ind, f"indicators missing fields: {missing_ind}"
        for key, val in indicators.items():
            assert val is None, f"indicators.{key} must be None when not analyzed"

        probs = data.get("probabilities")
        assert isinstance(probs, dict), "probabilities must be a dict"
        assert set(probs.keys()) == {"REAL", "FAKE"}
        assert 0.0 <= probs["REAL"] <= 1.0
        assert 0.0 <= probs["FAKE"] <= 1.0
        assert abs((probs["REAL"] + probs["FAKE"]) - 1.0) < 1e-3
    else:
        assert data["framesAnalyzed"] == 1, "framesAnalyzed must be 1 for image"
        assert data["facesDetected"] is None, "facesDetected should be None for image classifier"
        assert data["artifactsDetected"] is None, "artifactsDetected should be None for image classifier"
        assert data["demoMode"] is False, "demoMode should be False for real ML model"

        # Forensic indicators are not produced by the ResNet50 model
        indicators = data["indicators"]
        assert isinstance(indicators, dict), "indicators must be a dict"
        missing_ind = _REQUIRED_INDICATOR_FIELDS - indicators.keys()
        assert not missing_ind, f"indicators missing fields: {missing_ind}"
        for key, val in indicators.items():
            assert val is None, f"indicators.{key} must be None when not analyzed"

    # ── note: non-empty string ────────────────────────────────────────────
    assert isinstance(data["note"], str) and len(data["note"]) > 0, (
        "note must be a non-empty string"
    )


# ===========================================================================
# 1. Health endpoint
# ===========================================================================

class TestHealth:
    def test_returns_200(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_status_field_is_ok(self):
        resp = client.get("/api/health")
        assert resp.json()["status"] == "ok"

    def test_service_name_present(self):
        resp = client.get("/api/health")
        assert "service" in resp.json()

    def test_version_present(self):
        resp = client.get("/api/health")
        assert "version" in resp.json()


# ===========================================================================
# 2. Image detection — successful uploads
# ===========================================================================

class TestImageDetectionSuccess:
    def test_jpeg_returns_200(self):
        resp = client.post(
            "/api/detect/image",
            files={"file": ("photo.jpg", io.BytesIO(JPEG_BYTES), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_png_returns_200(self):
        resp = client.post(
            "/api/detect/image",
            files={"file": ("photo.png", io.BytesIO(PNG_BYTES), "image/png")},
        )
        assert resp.status_code == 200

    def test_webp_returns_200(self):
        resp = client.post(
            "/api/detect/image",
            files={"file": ("photo.webp", io.BytesIO(WEBP_BYTES), "image/webp")},
        )
        assert resp.status_code == 200

    def test_response_contains_all_required_fields(self):
        resp = client.post(
            "/api/detect/image",
            files={"file": ("photo.jpg", io.BytesIO(JPEG_BYTES), "image/jpeg")},
        )
        assert resp.status_code == 200
        _assert_valid_detection_response(resp.json(), expected_media_type="image")

    def test_media_type_is_image(self):
        resp = client.post(
            "/api/detect/image",
            files={"file": ("photo.jpg", io.BytesIO(JPEG_BYTES), "image/jpeg")},
        )
        assert resp.json()["mediaType"] == "image"

    def test_frames_analyzed_is_1_for_images(self):
        resp = client.post(
            "/api/detect/image",
            files={"file": ("photo.jpg", io.BytesIO(JPEG_BYTES), "image/jpeg")},
        )
        assert resp.json()["framesAnalyzed"] == 1

    def test_confidence_is_within_0_to_100(self):
        resp = client.post(
            "/api/detect/image",
            files={"file": ("photo.jpg", io.BytesIO(JPEG_BYTES), "image/jpeg")},
        )
        confidence = resp.json()["confidence"]
        assert 0.0 <= confidence <= 100.0

    def test_indicators_are_none_for_resnet_image(self):
        resp = client.post(
            "/api/detect/image",
            files={"file": ("photo.jpg", io.BytesIO(JPEG_BYTES), "image/jpeg")},
        )
        for key, val in resp.json()["indicators"].items():
            assert val is None, f"indicators.{key} must be None when not analyzed"

    def test_demo_mode_is_false(self):
        resp = client.post(
            "/api/detect/image",
            files={"file": ("photo.jpg", io.BytesIO(JPEG_BYTES), "image/jpeg")},
        )
        assert resp.json()["demoMode"] is False

    def test_corrupt_image_returns_422_invalid_image(self):
        resp = client.post(
            "/api/detect/image",
            files={"file": ("corrupt.jpg", io.BytesIO(CORRUPT_JPEG_BYTES), "image/jpeg")},
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail["status"] == "error"
        assert detail["error"] == "INVALID_IMAGE"
        assert "corrupt or cannot be decoded" in detail["message"]

    def test_truncated_image_returns_422_invalid_image(self):
        with open("image_processing/test_images/test.jpeg", "rb") as fh:
            real_bytes = fh.read()
        truncated_bytes = real_bytes[:500]
        resp = client.post(
            "/api/detect/image",
            files={"file": ("truncated.jpg", io.BytesIO(truncated_bytes), "image/jpeg")},
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail["status"] == "error"
        assert detail["error"] == "INVALID_IMAGE"
        assert "corrupt or cannot be decoded" in detail["message"]

    def test_prediction_is_a_valid_label(self):
        resp = client.post(
            "/api/detect/image",
            files={"file": ("photo.jpg", io.BytesIO(JPEG_BYTES), "image/jpeg")},
        )
        assert resp.json()["prediction"] in _VALID_PREDICTIONS

    def test_assessment_tone_matches_prediction(self):
        resp = client.post(
            "/api/detect/image",
            files={"file": ("photo.jpg", io.BytesIO(JPEG_BYTES), "image/jpeg")},
        )
        data = resp.json()
        expected_tone = {
            "FAKE":         "danger",
            "REAL":         "safe",
            "INCONCLUSIVE": "amber",
        }[data["prediction"]]
        assert data["assessmentTone"] == expected_tone


# ===========================================================================
# 3. Video detection — successful uploads
# ===========================================================================

class TestVideoDetectionSuccess:
    def test_mp4_returns_200(self):
        resp = client.post(
            "/api/detect/video",
            files={"file": ("clip.mp4", io.BytesIO(MP4_BYTES), "video/mp4")},
        )
        assert resp.status_code == 200

    def test_mov_returns_200(self):
        resp = client.post(
            "/api/detect/video",
            files={"file": ("clip.mov", io.BytesIO(MOV_BYTES), "video/quicktime")},
        )
        assert resp.status_code == 200

    def test_avi_returns_200(self):
        resp = client.post(
            "/api/detect/video",
            files={"file": ("clip.avi", io.BytesIO(AVI_BYTES), "video/x-msvideo")},
        )
        assert resp.status_code == 200

    def test_response_contains_all_required_fields(self):
        resp = client.post(
            "/api/detect/video",
            files={"file": ("clip.mp4", io.BytesIO(MP4_BYTES), "video/mp4")},
        )
        assert resp.status_code == 200
        _assert_valid_detection_response(resp.json(), expected_media_type="video")

    def test_media_type_is_video(self):
        resp = client.post(
            "/api/detect/video",
            files={"file": ("clip.mp4", io.BytesIO(MP4_BYTES), "video/mp4")},
        )
        assert resp.json()["mediaType"] == "video"

    def test_frames_analyzed_is_greater_than_zero(self):
        resp = client.post(
            "/api/detect/video",
            files={"file": ("clip.mp4", io.BytesIO(MP4_BYTES), "video/mp4")},
        )
        assert resp.json()["framesAnalyzed"] > 0

    def test_confidence_is_within_0_to_100(self):
        resp = client.post(
            "/api/detect/video",
            files={"file": ("clip.mp4", io.BytesIO(MP4_BYTES), "video/mp4")},
        )
        assert 0.0 <= resp.json()["confidence"] <= 100.0

    def test_indicators_are_none_for_fakestormer_video(self):
        resp = client.post(
            "/api/detect/video",
            files={"file": ("clip.mp4", io.BytesIO(MP4_BYTES), "video/mp4")},
        )
        for key, val in resp.json()["indicators"].items():
            assert val is None, f"indicators.{key} must be None when not analyzed"

    def test_demo_mode_is_false(self):
        resp = client.post(
            "/api/detect/video",
            files={"file": ("clip.mp4", io.BytesIO(MP4_BYTES), "video/mp4")},
        )
        assert resp.json()["demoMode"] is False

    def test_corrupt_video_returns_422_invalid_video(self):
        resp = client.post(
            "/api/detect/video",
            files={"file": ("corrupt.mp4", io.BytesIO(CORRUPT_MP4_BYTES), "video/mp4")},
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert detail["status"] == "error"
        assert detail["error"] == "INVALID_VIDEO"


# ===========================================================================
# 4. Missing file
# ===========================================================================

class TestMissingFile:
    def test_image_endpoint_without_file_returns_422(self):
        # FastAPI returns 422 Unprocessable Entity when a required File() field
        # is absent from the multipart body.
        resp = client.post("/api/detect/image")
        assert resp.status_code == 422

    def test_video_endpoint_without_file_returns_422(self):
        resp = client.post("/api/detect/video")
        assert resp.status_code == 422


# ===========================================================================
# 5. Unsupported file format
# ===========================================================================

class TestUnsupportedFormat:
    def test_pdf_to_image_endpoint_returns_415(self):
        resp = client.post(
            "/api/detect/image",
            files={"file": ("doc.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )
        assert resp.status_code == 415

    def test_pdf_to_video_endpoint_returns_415(self):
        resp = client.post(
            "/api/detect/video",
            files={"file": ("doc.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )
        assert resp.status_code == 415

    def test_video_file_rejected_by_image_endpoint(self):
        # Sending an MP4 to /api/detect/image should be rejected (wrong category).
        resp = client.post(
            "/api/detect/image",
            files={"file": ("clip.mp4", io.BytesIO(MP4_BYTES), "video/mp4")},
        )
        assert resp.status_code == 415

    def test_image_file_rejected_by_video_endpoint(self):
        # Sending a JPEG to /api/detect/video should be rejected.
        resp = client.post(
            "/api/detect/video",
            files={"file": ("photo.jpg", io.BytesIO(JPEG_BYTES), "image/jpeg")},
        )
        assert resp.status_code == 415

    def test_unsupported_format_error_has_correct_code(self):
        resp = client.post(
            "/api/detect/image",
            files={"file": ("doc.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )
        detail = resp.json()["detail"]
        assert detail["status"] == "error"
        assert detail["error"] == "UNSUPPORTED_FORMAT"

    def test_unsupported_format_error_includes_message(self):
        resp = client.post(
            "/api/detect/image",
            files={"file": ("doc.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )
        assert "message" in resp.json()["detail"]


# ===========================================================================
# 6. Oversized upload
# ===========================================================================

class TestOversizedUpload:
    """
    Instead of creating a real 100 MB file, we monkeypatch MAX_FILE_SIZE_BYTES
    in the file_validation module to a tiny value (10 bytes).  The JPEG_BYTES
    payload is 132 bytes, so it will always exceed the patched limit.
    """

    _TINY_LIMIT = 10  # bytes — intentionally below the size of any test payload

    def test_oversized_image_returns_413(self):
        with patch("backend.utils.file_validation.MAX_FILE_SIZE_BYTES", self._TINY_LIMIT):
            resp = client.post(
                "/api/detect/image",
                files={"file": ("photo.jpg", io.BytesIO(JPEG_BYTES), "image/jpeg")},
            )
        assert resp.status_code == 413

    def test_oversized_image_error_has_correct_code(self):
        with patch("backend.utils.file_validation.MAX_FILE_SIZE_BYTES", self._TINY_LIMIT):
            resp = client.post(
                "/api/detect/image",
                files={"file": ("photo.jpg", io.BytesIO(JPEG_BYTES), "image/jpeg")},
            )
        detail = resp.json()["detail"]
        assert detail["status"] == "error"
        assert detail["error"] == "FILE_TOO_LARGE"

    def test_oversized_video_returns_413(self):
        with patch("backend.utils.file_validation.MAX_FILE_SIZE_BYTES", self._TINY_LIMIT):
            resp = client.post(
                "/api/detect/video",
                files={"file": ("clip.mp4", io.BytesIO(MP4_BYTES), "video/mp4")},
            )
        assert resp.status_code == 413


# ===========================================================================
# 7. Processing service failure
# ===========================================================================

class TestServiceFailure:
    def test_image_service_error_returns_500(self):
        with patch(
            "backend.routers.detect.process_image",
            side_effect=ImageServiceError("Simulated inference failure"),
        ):
            resp = client.post(
                "/api/detect/image",
                files={"file": ("photo.jpg", io.BytesIO(JPEG_BYTES), "image/jpeg")},
            )
        assert resp.status_code == 500

    def test_image_service_error_response_structure(self):
        with patch(
            "backend.routers.detect.process_image",
            side_effect=ImageServiceError("Simulated inference failure"),
        ):
            resp = client.post(
                "/api/detect/image",
                files={"file": ("photo.jpg", io.BytesIO(JPEG_BYTES), "image/jpeg")},
            )
        detail = resp.json()["detail"]
        assert detail["status"] == "error"
        assert detail["error"] == "PROCESSING_FAILED"
        assert "message" in detail

    def test_video_service_error_returns_500(self):
        with patch(
            "backend.routers.detect.process_video",
            side_effect=VideoServiceError("Simulated pipeline failure"),
        ):
            resp = client.post(
                "/api/detect/video",
                files={"file": ("clip.mp4", io.BytesIO(MP4_BYTES), "video/mp4")},
            )
        assert resp.status_code == 500

    def test_video_service_error_response_structure(self):
        with patch(
            "backend.routers.detect.process_video",
            side_effect=VideoServiceError("Simulated pipeline failure"),
        ):
            resp = client.post(
                "/api/detect/video",
                files={"file": ("clip.mp4", io.BytesIO(MP4_BYTES), "video/mp4")},
            )
        detail = resp.json()["detail"]
        assert detail["status"] == "error"
        assert detail["error"] == "PROCESSING_FAILED"


# ===========================================================================
# 8. Temp file cleanup
# ===========================================================================

class TestTempFileCleanup:
    """Verify that temporary upload files are removed after processing,
    including when the service raises an exception."""

    def test_temp_file_removed_after_successful_image_analysis(self):
        saved: list[str] = []

        def _spy(path: str) -> dict:
            saved.append(path)
            # Call the real service to get a valid result
            from backend.services.image_service import process_image as _real
            return _real(path)

        with patch("backend.routers.detect.process_image", side_effect=_spy):
            client.post(
                "/api/detect/image",
                files={"file": ("photo.jpg", io.BytesIO(JPEG_BYTES), "image/jpeg")},
            )

        assert saved, "Spy was never called — temp path was not captured"
        for p in saved:
            assert not os.path.exists(p), f"Temp file not cleaned up: {p}"

    def test_temp_file_removed_after_image_service_error(self):
        saved: list[str] = []

        def _failing(path: str) -> dict:
            saved.append(path)
            raise ImageServiceError("Simulated failure")

        with patch("backend.routers.detect.process_image", side_effect=_failing):
            client.post(
                "/api/detect/image",
                files={"file": ("photo.jpg", io.BytesIO(JPEG_BYTES), "image/jpeg")},
            )

        assert saved, "Spy was never called — temp path was not captured"
        for p in saved:
            assert not os.path.exists(p), f"Temp file not cleaned up after error: {p}"

    def test_temp_file_removed_after_invalid_image_error(self):
        saved: list[str] = []

        def _spy(path: str) -> dict:
            saved.append(path)
            from backend.services.image_service import process_image as _real
            return _real(path)

        with patch("backend.routers.detect.process_image", side_effect=_spy):
            resp = client.post(
                "/api/detect/image",
                files={"file": ("corrupt.jpg", io.BytesIO(CORRUPT_JPEG_BYTES), "image/jpeg")},
            )
            assert resp.status_code == 422

        assert saved, "Spy was never called — temp path was not captured"
        for p in saved:
            assert not os.path.exists(p), f"Temp file not cleaned up after invalid image: {p}"

    def test_temp_file_removed_after_successful_video_analysis(self):
        saved: list[str] = []

        def _spy(path: str) -> dict:
            saved.append(path)
            from backend.services.video_service import process_video as _real
            return _real(path)

        with patch("backend.routers.detect.process_video", side_effect=_spy):
            client.post(
                "/api/detect/video",
                files={"file": ("clip.mp4", io.BytesIO(MP4_BYTES), "video/mp4")},
            )

        assert saved
        for p in saved:
            assert not os.path.exists(p), f"Temp file not cleaned up: {p}"

    def test_temp_file_removed_after_invalid_video_error(self):
        saved: list[str] = []

        def _spy(path: str) -> dict:
            saved.append(path)
            from backend.services.video_service import process_video as _real
            return _real(path)

        with patch("backend.routers.detect.process_video", side_effect=_spy):
            resp = client.post(
                "/api/detect/video",
                files={"file": ("corrupt.mp4", io.BytesIO(CORRUPT_MP4_BYTES), "video/mp4")},
            )
            assert resp.status_code == 422

        assert saved
        for p in saved:
            assert not os.path.exists(p), f"Temp file not cleaned up after invalid video: {p}"


# ===========================================================================
# 9. Real model smoke test
# ===========================================================================

class TestModelInferenceSmoke:
    """
    End-to-end smoke test verifying that the real trained model on disk
    runs inference successfully via the API and returns a valid contract.
    Does not assert specific predictions or artificial confidence thresholds.
    """

    def test_sample_image_inference_returns_valid_contract(self):
        sample_path = "image_processing/test_images/test.jpeg"
        if not os.path.isfile(sample_path):
            return

        with open(sample_path, "rb") as fh:
            real_bytes = fh.read()

        resp = client.post(
            "/api/detect/image",
            files={"file": ("test.jpeg", io.BytesIO(real_bytes), "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        _assert_valid_detection_response(data, expected_media_type="image")
        assert data["demoMode"] is False
        assert data["prediction"] in _VALID_PREDICTIONS
        assert 0.0 <= data["confidence"] <= 100.0
        assert data["processingTime"] > 0
