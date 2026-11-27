"""
File upload validation utilities.

Validates file type using magic-byte detection (not the client-provided
Content-Type header) and enforces the 100 MB upload size limit.

Supported formats:
    Images  — JPEG, PNG, WebP
    Videos  — MP4, MOV (QuickTime), AVI
"""

from __future__ import annotations

from fastapi import HTTPException, UploadFile, status

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FILE_SIZE_BYTES: int = 100 * 1024 * 1024  # 100 MB

ALLOWED_IMAGE_MIME_TYPES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/webp",
})

ALLOWED_VIDEO_MIME_TYPES: frozenset[str] = frozenset({
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
})

ALLOWED_MIME_TYPES: frozenset[str] = ALLOWED_IMAGE_MIME_TYPES | ALLOWED_VIDEO_MIME_TYPES

_SUPPORTED_FORMATS_MSG = (
    "JPEG, PNG, WebP (images) and MP4, MOV, AVI (videos)"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_mime_from_header(header: bytes) -> str | None:
    """
    Detect the MIME type of a file from its leading magic bytes.

    Reads at most the first 12 bytes. Returns None for any unrecognised
    byte sequence so callers can return a 415 response.

    Supported signatures:
        JPEG    FF D8 FF  (first 3 bytes)
        PNG     89 50 4E 47 0D 0A 1A 0A  (first 8 bytes)
        WebP    RIFF....WEBP  (bytes 0–3 and 8–11)
        AVI     RIFF....AVI   (bytes 0–3 and 8–11)
        MP4     ....ftyp<brand>  (bytes 4–7 = "ftyp")
        MOV     ....ftypqt      (bytes 4–7 = "ftyp", brand = "qt  ")
    """
    # JPEG — magic: FF D8 FF
    if header[:3] == b"\xff\xd8\xff":
        return "image/jpeg"

    # PNG — magic: 89 50 4E 47 0D 0A 1A 0A
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"

    # RIFF container: WebP and AVI both open with "RIFF"
    if len(header) >= 12 and header[:4] == b"RIFF":
        subtype = header[8:12]
        if subtype == b"WEBP":
            return "image/webp"
        if subtype == b"AVI ":
            return "video/x-msvideo"

    # ISO Base Media File Format — bytes 4–7 are "ftyp" for MP4 and MOV
    if len(header) >= 8 and header[4:8] == b"ftyp":
        brand = header[8:12] if len(header) >= 12 else b""
        if brand == b"qt  ":
            # QuickTime / MOV
            return "video/quicktime"
        # All other ftyp brands (isom, mp41, mp42, avc1, …) treated as MP4
        return "video/mp4"

    return None


def _error_detail(code: str, message: str) -> dict:
    """Build a consistent error detail dict for HTTPException."""
    return {"status": "error", "error": code, "message": message}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def read_and_validate_upload(
    upload: UploadFile,
    allowed_mime_types: frozenset[str] | None = None,
) -> tuple[bytes, str]:
    """
    Read an uploaded file, validate its magic bytes and enforce the size limit.

    Args:
        upload:            FastAPI UploadFile from the request.
        allowed_mime_types: Restrict to a specific subset of MIME types.
                            Defaults to all supported types.

    Returns:
        (content, detected_mime) — the raw file bytes and detected MIME type.

    Raises:
        HTTPException 400  — file missing or too small to validate.
        HTTPException 413  — file exceeds MAX_FILE_SIZE_BYTES.
        HTTPException 415  — file type is not in allowed_mime_types.
    """
    if upload is None or not upload.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_detail("MISSING_FILE", "No file was provided."),
        )

    allowed = allowed_mime_types if allowed_mime_types is not None else ALLOWED_MIME_TYPES

    # Read the first 12 bytes to detect the file type from magic bytes.
    # 12 bytes covers all signatures we need to check.
    header = await upload.read(12)

    if len(header) < 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_error_detail(
                "INVALID_FILE",
                "The uploaded file is too small to be a valid image or video.",
            ),
        )

    detected_mime = _detect_mime_from_header(header)

    if detected_mime is None or detected_mime not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=_error_detail(
                "UNSUPPORTED_FORMAT",
                f"The file format is not supported. "
                f"Accepted formats: {_SUPPORTED_FORMATS_MSG}.",
            ),
        )

    # Read the remainder of the file.
    # Request one extra byte beyond the limit so we can detect oversized uploads
    # without reading the entire file into memory first.
    remaining_limit = max(0, MAX_FILE_SIZE_BYTES - len(header) + 1)
    rest = await upload.read(remaining_limit)

    total_size = len(header) + len(rest)
    if total_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=_error_detail(
                "FILE_TOO_LARGE",
                f"The file exceeds the 100 MB upload limit "
                f"(received approximately {total_size // (1024 * 1024)} MB).",
            ),
        )

    return header + rest, detected_mime
