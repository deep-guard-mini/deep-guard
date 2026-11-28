
"""
DeepGuard image processing service.

This service connects the FastAPI backend to our trained image
classification model.

Flow:
    Uploaded image
        ↓
    predict_image()
        ↓
    Trained ResNet50 model
        ↓
    REAL / FAKE prediction
        ↓
    Service result returned to backend
"""

# Allows type hints such as str | Path
from __future__ import annotations

# Path helps us accept file paths in a platform-independent way.
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# Import the prediction function from our ML package.
# This function loads the image and sends it through our trained model.
from image_processing.predictor import predict_image


class ImageServiceError(Exception):
    """
    Custom exception for image processing failures.

    We use this exception to distinguish errors in our image
    inference pipeline from normal application errors.
    """
    pass


class InvalidImageError(Exception):
    """
    Raised when an uploaded image is corrupt, truncated, or cannot be decoded.
    """
    pass


def validate_image_file(image_path: str | Path) -> None:
    """
    Verify that the uploaded image can be opened and decoded before inference.

    Raises:
        InvalidImageError: If the image is corrupt, truncated, or unreadable.
    """
    path = Path(image_path)
    try:
        with Image.open(path) as img:
            img.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(
            f"Uploaded image is corrupt or cannot be decoded: {exc}"
        ) from exc


def process_image(image_path: str | Path) -> dict:
    """
    Analyze an image using DeepGuard's trained ML model.

    Args:
        image_path:
            Path to the uploaded image saved temporarily on disk.

    Returns:
        Dictionary containing:
            prediction
            confidence
            probabilities
            processing_time
            faces_detected
            frames_analyzed
            artifacts_detected
            indicators
            note
            demo_mode
    """

    # Pre-validate that the uploaded image can actually be decoded by Pillow
    # before invoking the ML model.
    validate_image_file(image_path)

    try:
        # --------------------------------------------------
        # STEP 1: RUN REAL ML INFERENCE
        # --------------------------------------------------
        # Call the actual trained model.
        #
        # predict_image() handles:
        #   - Reading the image
        #   - Converting it to RGB
        #   - Resizing to the model's expected dimensions
        #   - Normalizing the image
        #   - Running ResNet50 inference
        #   - Returning the classification result
        #
        # This is no longer random/mock prediction.
        raw = predict_image(image_path)


        # --------------------------------------------------
        # STEP 2: EXTRACT MODEL OUTPUTS
        # --------------------------------------------------

        # Predicted class: REAL or FAKE
        prediction = raw["prediction"]

        # Confidence score from 0 to 100
        confidence = float(raw["confidence"])

        # Actual inference duration in seconds
        processing_time = float(raw["processing_time"])

        # Probability scores for both classes.
        # Example:
        # {"REAL": 0.9738, "FAKE": 0.0262}
        probabilities = raw["probabilities"]


        # --------------------------------------------------
        # STEP 3: BUILD SERVICE RESPONSE
        # --------------------------------------------------
        # We return the fields expected by our response builder.
        #
        # Important:
        # Our current ML model only performs image classification.
        # It does NOT perform:
        #   - Face detection/counting
        #   - Artifact localization
        #   - Lighting analysis
        #   - Texture forensic scoring
        #
        # Therefore, unsupported metrics are None (JSON null),
        # not fabricated numbers.

        return {
            # Actual model prediction
            "prediction": prediction,

            # Actual model confidence
            "confidence": confidence,

            # Face detection is not implemented in this model
            "faces_detected": None,

            # One uploaded still image is being analyzed
            "frames_analyzed": 1,

            # Artifact detection is not implemented
            "artifacts_detected": None,

            # Actual time spent on inference
            "processing_time": processing_time,

            # Forensic indicators are not produced by ResNet50
            "indicators": {
                "face_consistency": None,
                "texture_consistency": None,
                "lighting_consistency": None,
                "compression_artifacts": None,
                "edge_artifacts": None,
            },

            # REAL and FAKE probability scores
            "probabilities": probabilities,

            # Human-readable explanation for the frontend
            "note": (
                f"DeepGuard's trained image classification model "
                f"predicted {prediction} with {confidence:.2f}% "
                f"confidence. This result is an AI estimate and "
                f"does not guarantee image authenticity."
            ),

            # False because we're using our trained model
            "demo_mode": False,
        }

    except Exception as exc:
        # If model inference fails, wrap the original error
        # in our service-specific exception.
        #
        # 'from exc' preserves the original exception chain,
        # which is useful when debugging.
        raise ImageServiceError(
            f"Image inference failed: {exc}"
        ) from exc
