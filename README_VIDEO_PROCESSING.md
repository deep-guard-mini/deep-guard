# DeepGuard — Video Processing Handoff (Sanjeev)

This package contains the video-specific inference pipeline for the DeepGuard project. It is intended to be copied into the repository root and called by the backend developer.

## 1. What this module does

`video_processing/pipeline.py` exposes:

```python
from video_processing.pipeline import process_video
result = process_video("path/to/video.mp4")
```

Pipeline:

1. Opens the video with OpenCV.
2. Samples up to 10 frames evenly across the video.
3. Detects faces with OpenCV Haar Cascade (`scaleFactor=1.1`, `minNeighbors=5`, `minSize=(60, 60)`).
4. Converts each face crop to RGB, resizes it to 224x224, and applies ImageNet normalization.
5. Runs the trained ResNet-50 classifier (`real=0`, `fake=1`).
6. Averages fake-class probabilities across all detected face crops in that video.
7. Returns `FAKE` when mean fake probability is at least 0.5; otherwise returns `REAL`.
8. Returns `INCONCLUSIVE` if no faces are detected.

The implementation follows the Colab inference procedure used for the DeepGuard multi-manipulation checkpoint. It uses the checkpoint from epoch 5.

## 2. Files to copy

Copy this folder into the root of the `deep-guard-main` repository:

```text
video_processing/
    __init__.py
    pipeline.py
```

The importable Python package uses an underscore (`video_processing`). This is intentional: Python package names cannot contain a hyphen. It replaces the currently empty `video-processing/` placeholder folder in the supplied project ZIP. The team can rename/remove that empty placeholder after integration.

## 3. Model checkpoint is a separate file

The model weights are not included in this handoff ZIP. Copy the trained file:

```text
deepguard_multimanip_best.pth
```

into the repository:

```text
models/deepguard_multimanip_best.pth
```

Alternatively, set an environment variable before starting FastAPI:

**Windows PowerShell**

```powershell
$env:DEEPGUARD_MODEL_PATH="D:\DeepGuard\models\deepguard_multimanip_best.pth"
```

**Windows CMD**

```cmd
set DEEPGUARD_MODEL_PATH=D:\DeepGuard\models\deepguard_multimanip_best.pth
```

**Linux/macOS**

```bash
export DEEPGUARD_MODEL_PATH=/path/to/deepguard_multimanip_best.pth
```

Do not commit the large `.pth` checkpoint to ordinary Git unless the team explicitly chooses Git LFS or another model-artifact storage method. Share the checkpoint separately through the team's agreed Drive/storage location.

## 4. Install dependencies

From the repository root, in the same virtual environment used by FastAPI:

```bash
pip install -r requirements-video-processing.txt
```

If the backend already installs `opencv-python`, avoid installing both `opencv-python` and `opencv-python-headless` in the same environment. Keep one OpenCV distribution. On a server without a GUI, `opencv-python-headless` is suitable.

PyTorch automatically uses CUDA when available and otherwise runs on CPU. For GPU acceleration, install the PyTorch build compatible with the machine's CUDA/driver from the official PyTorch installation selector; do not blindly replace a working CUDA-enabled PyTorch installation.

## 5. Backend integration (for Vikas)

The current project has a mock implementation in `backend/services/video_service.py`. Replace the mock `process_video()` body with this adapter. The existing API route can continue calling the service as before.

```python
from pathlib import Path
from video_processing.pipeline import (
    VideoProcessingError,
    process_video as run_video_pipeline,
)

class VideoServiceError(Exception):
    """Raised when the real video pipeline fails."""


def process_video(video_path: str | Path) -> dict:
    try:
        return run_video_pipeline(video_path)
    except VideoProcessingError as exc:
        raise VideoServiceError(str(exc)) from exc
```

The returned dictionary contains the keys required by the existing `backend/utils/response_builder.py`:

- `prediction`: `REAL`, `FAKE`, or `INCONCLUSIVE`
- `confidence`: predicted-class score from 0 to 100
- `faces_detected`: number of face crops found across sampled frames
- `frames_analyzed`: number of sampled frames successfully decoded
- `artifacts_detected`: `0` (not computed by this classifier)
- `processing_time`: seconds
- `indicators`: the five schema-compatible values, currently `0` because the ResNet-50 pipeline does not compute these forensic measurements
- `note`: explanation of the inference and its limitations
- `demo_mode`: `False`

The module also returns `fake_probability` and `real_probability` as percentages. The current response builder does not forward these extra fields; the backend developer can add them to the API response if the frontend needs both scores.

### Important UI/contract note

The current frontend response schema contains forensic indicator fields. This model only performs face-crop classification and probability aggregation; it does **not** calculate face consistency, texture consistency, lighting consistency, compression artifacts, or edge artifacts. The zeros returned for those fields are placeholders for schema compatibility, not measured scores. The backend/frontend should label them unavailable or hide those indicators rather than presenting zero as a forensic finding.

## 6. Direct smoke test

After placing the checkpoint and installing dependencies, run from the repository root:

```bash
python -c "from video_processing.pipeline import process_video; print(process_video('sample.mp4'))"
```

Expected output shape (values depend on the input video):

```python
{
    "prediction": "REAL",  # or "FAKE" / "INCONCLUSIVE"
    "confidence": 85.88,
    "fake_probability": 14.12,
    "real_probability": 85.88,
    "faces_detected": 12,
    "frames_analyzed": 10,
    "artifacts_detected": 0,
    "processing_time": 1.23,
    "indicators": {
        "face_consistency": 0,
        "texture_consistency": 0,
        "lighting_consistency": 0,
        "compression_artifacts": 0,
        "edge_artifacts": 0
    },
    "note": "...",
    "demo_mode": False
}
```

## 7. Important interpretation limits

- The mean fake probability is an average model score, not calibrated certainty.
- The default threshold is 0.5, matching the Colab evaluation. Do not change it during integration without a separately documented validation experiment.
- `faces_detected` counts face crops, not unique people tracked across the clip.
- The model was trained on FaceForensics++ C23 manipulations (Deepfakes, Face2Face, FaceSwap). The independent 150-video evaluation achieved 87.33% video-level accuracy, balanced accuracy 84.00%, and ROC-AUC 0.9148 on unseen source identities within the same dataset family. This does not establish performance on arbitrary real-world videos or unrelated datasets.
- No-face videos return `INCONCLUSIVE`; the backend should handle this as a valid result, not as a server failure.

## 8. Ownership boundary

This package owns video decoding, frame sampling, face detection/cropping, preprocessing, model invocation for sampled face crops, and video-level probability aggregation. The backend owns HTTP upload validation, temporary-file lifecycle, API response formatting, and error translation. The frontend owns upload UI and display of the API response.
