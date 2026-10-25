"""
Response builder.

Maps the raw dict returned by image_service.process_image() or
video_service.process_video() into the full JSON response shape that
the DeepGuard frontend expects.

All camelCase field names and value conventions match those used in:
    frontend/src/services/mockAnalysis.js
"""

from __future__ import annotations

# Maps ML prediction labels → (human-readable assessment, CSS tone class)
_ASSESSMENT_MAP: dict[str, tuple[str, str]] = {
    "FAKE":         ("LIKELY MANIPULATED", "danger"),
    "REAL":         ("LIKELY AUTHENTIC",   "safe"),
    "INCONCLUSIVE": ("INCONCLUSIVE",       "amber"),
}


def build_detection_response(service_result: dict, media_type: str) -> dict:
    """
    Build the full API response from a processing service result dict.

    Args:
        service_result: Dict returned by process_image() or process_video().
                        See each service module for the exact key contract.
        media_type:     "image" or "video".

    Returns:
        A dict containing every field required by the frontend:
            status, prediction, confidence, assessment, assessmentTone,
            mediaType, processingTime, facesDetected, framesAnalyzed,
            artifactsDetected, indicators (5 sub-fields), note, demoMode.
    """
    prediction: str = service_result.get("prediction", "INCONCLUSIVE")
    confidence: float = float(service_result.get("confidence", 0.0))

    assessment, assessment_tone = _ASSESSMENT_MAP.get(
        prediction, ("INCONCLUSIVE", "amber")
    )

    raw_indicators: dict = service_result.get("indicators", {})

    return {
        "status":            "success",
        "prediction":        prediction,
        "confidence":        round(confidence, 1),
        "assessment":        assessment,
        "assessmentTone":    assessment_tone,
        "mediaType":         media_type,
        "processingTime":    round(float(service_result.get("processing_time", 0.0)), 2),
        "facesDetected":     int(service_result.get("faces_detected", 0)),
        "framesAnalyzed":    int(service_result.get("frames_analyzed", 1)),
        "artifactsDetected": int(service_result.get("artifacts_detected", 0)),
        "indicators": {
            "faceConsistency":      int(raw_indicators.get("face_consistency", 0)),
            "textureConsistency":   int(raw_indicators.get("texture_consistency", 0)),
            "lightingConsistency":  int(raw_indicators.get("lighting_consistency", 0)),
            "compressionArtifacts": int(raw_indicators.get("compression_artifacts", 0)),
            "edgeArtifacts":        int(raw_indicators.get("edge_artifacts", 0)),
        },
        "note":     service_result.get("note", ""),
        "demoMode": bool(service_result.get("demo_mode", False)),
    }
