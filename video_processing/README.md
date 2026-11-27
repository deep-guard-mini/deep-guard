# DeepGuard — Video Processing Module (FakeSTormer C23)

## Overview

This module performs video-level deepfake detection using the trained **FakeSTormer** (`TopDownDetector` with `TimeViT` divided space-time backbone) model trained on FaceForensics++ C23.

Module owner: Video Processing

This module contains no FastAPI routes or frontend logic.

## Files

- `video_loader.py` — Evenly spaced video frame sampling, Haar Cascade face cropping (`scaleFactor=1.1`, `minNeighbors=5`, `minSize=(60, 60)`, `pad_ratio=0.25`), and center-crop fallback when no face is detected
- `fakestormer_inference.py` — FakeSTormer config/checkpoint loading, 224x224 ImageNet preprocessing, 4-clip temporal aggregation (`16` frames total), and CPU/CUDA inference
- `predictor.py` — Cached model loader and path-based inference interface (`predict_video` / `predict_video_file`)
- `FakeSTormer_repo/` — Minimal runtime model, config, and registry modules required by FakeSTormer

## Model & Configuration

- Architecture: `TopDownDetector` (`TimeViT` + `TopdownHeatmapSimpleHead`)
- Config: `video_processing/FakeSTormer_repo/configs/temporal/FakeSFormer_base_c23.yaml`
- Framework: PyTorch + MMEngine + timm
- Input size: `4` clips × `4` frames = `16` frames per video, resized to `224 x 224` RGB and normalized with ImageNet mean/std (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`)
- Decision threshold: `0.5` on averaged clip fake probability (`>= 0.5` → `FAKE`, `< 0.5` → `REAL`)
- Device: Uses CUDA automatically when available, otherwise runs on CPU

## Integration interface

```python
from video_processing.predictor import predict_video

result = predict_video("path/to/video.mp4")
```

Example output:

```json
{
    "prediction": "FAKE",
    "label": "FAKE",
    "confidence": 80.01,
    "real_probability": 0.1999,
    "fake_probability": 0.8001,
    "probabilities": {
        "REAL": 0.1999,
        "FAKE": 0.8001
    },
    "frames_analyzed": 16,
    "faces_detected": 16,
    "processing_time": 4.215
}
```

## Model checkpoint

The trained FakeSTormer C23 checkpoint (`~943 MB`) exceeds GitHub's 100 MB file size limit and is excluded from Git via `.gitignore` (`*.pth`).

Place the trained checkpoint at either:

- `models/fakestormer_c23_best.pth`, or
- `models/TopDownDetector_C23_ViTBase224_ST_hm100_tempLOC0.2_4SBI_SAM_mp0.01_temp2_0vidAug_0.35_temp_normlms_normREAL_model_best.pth`

Alternatively, configure one of these environment variables:

```powershell
$env:FAKESTORMER_MODEL_PATH="C:\path\to\fakestormer_c23_best.pth"
# Or auto-download on startup if missing:
$env:FAKESTORMER_WEIGHTS_URL="https://..."
```
