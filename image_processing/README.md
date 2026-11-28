# DeepGuard — Image Processing Module

## Overview

This module performs binary deepfake image classification
using an ImageNet-pretrained ResNet50 model fine-tuned
on the HiDF dataset.

Module owner: Image Processing / ML

This module contains no FastAPI routes or frontend logic.

## Files

- `image_loader.py` — Basic PIL image loading
- `deepguard_inference.py` — ResNet50 model loading and inference
- `predictor.py` — Path-based integration interface

## Model

Architecture: ResNet50
Framework: PyTorch
Input size: 224 x 224 RGB
Classes: REAL (0), FAKE (1)

Preprocessing:
1. Convert image to RGB
2. Resize to 224 x 224
3. Convert to tensor
4. Normalize using ImageNet mean and standard deviation

## Final held-out test results

Dataset: HiDF
Test images: 10,713

- Accuracy: 85.45%
- FAKE precision: 83.13%
- FAKE recall: 86.52%
- FAKE F1-score: 84.79%
- ROC-AUC: 0.9238

These results describe performance on the held-out HiDF
test partition, not guaranteed real-world accuracy.

## Integration interface

Function:

    predict_image(image_path: str | Path) -> dict

Example:

    {
        "prediction": "REAL",
        "confidence": 64.26,
        "probabilities": {
            "REAL": 0.6426,
            "FAKE": 0.3574
        },
        "processing_time": 0.665
    }

`confidence` is a softmax score scaled to 0–100,
not a calibrated probability of correctness.

`probabilities` contains scores on the 0–1 scale.

## Model checkpoint

Expected default location:

    models/deepguard_resnet50_best.pth

Alternatively set the environment variable:

    DEEPGUARD_MODEL_PATH=/absolute/path/to/checkpoint.pth

The checkpoint is stored separately in Google Drive
and is not included in this module's source files.

## Integration notes for backend

- Backend owns API routes, upload validation and responses.
- Backend can call `predict_image(image_path)`.
- The model is loaded once per process and cached.
- The module performs classification, not face detection.
- It does not calculate individual forensic indicator scores.
- Do not fabricate unavailable forensic measurements.
- Backend must handle unavailable fields appropriately.
- The predictor reads the supplied file but does not
  create persistent copies of uploaded images.
- Backend owns temporary file creation and cleanup.

## Limitations

- Binary REAL/FAKE classification only.
- No calibrated confidence scores.
- No face count or localization.
- No individual manipulation artifact measurements.
- No video aggregation in this module.
- Performance on other datasets has not been established.
