
"""
DeepGuard response builder.

Converts the processing service output into a consistent
JSON response structure consumed by the frontend.
"""

from __future__ import annotations


# ----------------------------------------------------------
# STEP 1: MAP PREDICTIONS TO USER-FRIENDLY ASSESSMENTS
# ----------------------------------------------------------
# These values help the frontend display a readable assessment
# and a corresponding visual tone.

_ASSESSMENT_MAP: dict[str, tuple[str, str]] = {
    "FAKE": ("LIKELY MANIPULATED", "danger"),
    "REAL": ("LIKELY AUTHENTIC", "safe"),
    "INCONCLUSIVE": ("INCONCLUSIVE", "amber"),
}


# ----------------------------------------------------------
# STEP 2: HELPER FUNCTION FOR OPTIONAL INTEGER VALUES
# ----------------------------------------------------------
def _optional_int(value):
    """
    Convert a value to an integer if available.

    If the metric is unavailable (None), preserve it as None.

    Why?
    The current image model does not detect faces or forensic
    artifacts. We must not convert unavailable metrics to zero.

    None in Python becomes null in JSON.
    """

    if value is None:
        return None

    return int(value)


# ----------------------------------------------------------
# STEP 3: BUILD THE FINAL API RESPONSE
# ----------------------------------------------------------
def build_detection_response(
    service_result: dict,
    media_type: str
) -> dict:
    """
    Build the response returned by the DeepGuard API.

    Args:
        service_result:
            Dictionary received from image_service or video_service.

        media_type:
            Identifies the analyzed media type ("image" or "video").

    Returns:
        Frontend-compatible response dictionary.
    """

    # ------------------------------------------------------
    # STEP 3.1: EXTRACT MAIN PREDICTION
    # ------------------------------------------------------

    # If prediction is missing, default to INCONCLUSIVE.
    prediction = service_result.get(
        "prediction",
        "INCONCLUSIVE"
    )

    # Convert confidence into a numeric value.
    confidence = float(
        service_result.get("confidence", 0.0)
    )


    # ------------------------------------------------------
    # STEP 3.2: GET DISPLAY ASSESSMENT
    # ------------------------------------------------------

    # Example:
    # REAL → ("LIKELY AUTHENTIC", "safe")
    # FAKE → ("LIKELY MANIPULATED", "danger")
    #
    # Unknown predictions fall back to INCONCLUSIVE.
    assessment, assessment_tone = _ASSESSMENT_MAP.get(
        prediction,
        ("INCONCLUSIVE", "amber")
    )


    # ------------------------------------------------------
    # STEP 3.3: EXTRACT INDICATORS
    # ------------------------------------------------------

    # Some models may not return forensic indicators.
    # Using 'or {}' prevents errors if indicators is None.
    raw_indicators = service_result.get("indicators") or {}


    # ------------------------------------------------------
    # STEP 3.4: CREATE FINAL RESPONSE
    # ------------------------------------------------------

    return {

        # API request completed successfully
        "status": "success",

        # ML classification
        "prediction": prediction,

        # Confidence percentage, rounded to 1 decimal place
        "confidence": round(confidence, 1),

        # Frontend-friendly classification text
        "assessment": assessment,

        # Frontend styling category
        "assessmentTone": assessment_tone,

        # Image or video
        "mediaType": media_type,

        # Inference duration in seconds, rounded to 2 decimals
        "processingTime": round(
            float(
                service_result.get("processing_time", 0.0)
            ),
            2
        ),

        # Face count is unavailable when the service returns None
        "facesDetected": _optional_int(
            service_result.get("faces_detected")
        ),

        # Number of frames analyzed
        "framesAnalyzed": _optional_int(
            service_result.get("frames_analyzed", 1)
        ),

        # Artifact count may be unavailable
        "artifactsDetected": _optional_int(
            service_result.get("artifacts_detected")
        ),

        # --------------------------------------------------
        # FORENSIC INDICATORS
        # --------------------------------------------------
        # These fields are preserved for frontend compatibility.
        # If unavailable, they become None → JSON null.
        "indicators": {

            "faceConsistency": _optional_int(
                raw_indicators.get("face_consistency")
            ),

            "textureConsistency": _optional_int(
                raw_indicators.get("texture_consistency")
            ),

            "lightingConsistency": _optional_int(
                raw_indicators.get("lighting_consistency")
            ),

            "compressionArtifacts": _optional_int(
                raw_indicators.get("compression_artifacts")
            ),

            "edgeArtifacts": _optional_int(
                raw_indicators.get("edge_artifacts")
            ),
        },

        # Probability scores returned by our classifier.
        # These are optional because video service may not
        # return them yet.
        "probabilities": service_result.get("probabilities"),

        # Human-readable explanation
        "note": service_result.get("note", ""),

        # False means actual ML inference is active
        "demoMode": bool(
            service_result.get("demo_mode", False)
        ),
    }
