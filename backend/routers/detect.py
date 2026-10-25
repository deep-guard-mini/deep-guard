"""
Detection router.

Provides:
    POST /api/detect/image  — analyse an image for deepfake indicators
    POST /api/detect/video  — analyse a video for deepfake indicators

Both endpoints:
  1. Validate the upload (magic bytes + size limit) via file_validation.
  2. Save the file to a secure temporary path.
  3. Call the appropriate processing service (image_service / video_service).
  4. Build and return the standardised JSON response via response_builder.
  5. Clean up the temporary file in all cases, including error paths.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.services.image_service import ImageServiceError, process_image
from backend.services.video_service import VideoServiceError, process_video
from backend.utils.file_validation import (
    ALLOWED_IMAGE_MIME_TYPES,
    ALLOWED_VIDEO_MIME_TYPES,
    read_and_validate_upload,
)
from backend.utils.response_builder import build_detection_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/detect", tags=["detection"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ext_from_filename(filename: str | None, fallback: str) -> str:
    """Return the file extension from a filename, or a safe fallback."""
    if filename:
        suffix = Path(filename).suffix
        if suffix:
            return suffix
    return fallback


async def _validate_and_save(
    upload: UploadFile,
    allowed_mime_types: frozenset[str],
    fallback_ext: str,
) -> tuple[str, str]:
    """
    Validate the upload and write its bytes to a secure temporary file.

    Returns:
        (tmp_path, detected_mime)

    Raises:
        HTTPException for validation errors (propagated from read_and_validate_upload).
        RuntimeError if the temp file cannot be written (caught by caller).
    """
    content, detected_mime = await read_and_validate_upload(upload, allowed_mime_types)

    suffix = _ext_from_filename(upload.filename, fallback_ext)
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
    except Exception:
        # Ensure the fd / partial file is released before re-raising.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return tmp_path, detected_mime


def _cleanup(tmp_path: str) -> None:
    """Silently remove a temporary file, logging any OS errors."""
    try:
        os.unlink(tmp_path)
    except OSError as exc:
        logger.warning("Could not remove temp file %s: %s", tmp_path, exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/image", summary="Analyse an image for deepfake indicators")
async def detect_image(file: UploadFile = File(...)) -> dict:
    """
    Accepts a JPEG, PNG, or WebP image (max 100 MB) and returns deepfake
    detection results.

    **Request** — multipart/form-data, field name: `file`

    **Response** — JSON with prediction, confidence, forensic indicators
    and metadata. See `docs/TEAM_MODULE_CONTRACTS.md` for the full contract.
    """
    tmp_path, _ = await _validate_and_save(
        upload=file,
        allowed_mime_types=ALLOWED_IMAGE_MIME_TYPES,
        fallback_ext=".jpg",
    )

    try:
        service_result = process_image(tmp_path)
    except ImageServiceError as exc:
        logger.error("Image processing failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status":  "error",
                "error":   "PROCESSING_FAILED",
                "message": str(exc),
            },
        )
    finally:
        _cleanup(tmp_path)

    return build_detection_response(service_result, media_type="image")


@router.post("/video", summary="Analyse a video for deepfake indicators")
async def detect_video(file: UploadFile = File(...)) -> dict:
    """
    Accepts an MP4, MOV, or AVI video (max 100 MB) and returns deepfake
    detection results including per-frame analysis metadata.

    **Request** — multipart/form-data, field name: `file`

    **Response** — JSON with prediction, confidence, frames_analyzed,
    forensic indicators and metadata.
    """
    tmp_path, _ = await _validate_and_save(
        upload=file,
        allowed_mime_types=ALLOWED_VIDEO_MIME_TYPES,
        fallback_ext=".mp4",
    )

    try:
        service_result = process_video(tmp_path)
    except VideoServiceError as exc:
        logger.error("Video processing failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status":  "error",
                "error":   "PROCESSING_FAILED",
                "message": str(exc),
            },
        )
    finally:
        _cleanup(tmp_path)

    return build_detection_response(service_result, media_type="video")
