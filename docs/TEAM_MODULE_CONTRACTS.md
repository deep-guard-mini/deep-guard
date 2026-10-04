# DeepGuard — Team Module Contracts & Development Workflow

## Purpose

This document defines what each team member owns, what their module must accept, what it must return, and what the next module can expect.

The goal is to allow all four members to develop independently and integrate the modules later without rewriting everything.

---

# 1. Team Ownership

| Member | Role | Branch | Primary Folder |
|---|---|---|---|
| Yashas | Frontend | `feature/frontend` | `frontend/` |
| Vikas | Backend / API | `feature/backend` | `backend/` |
| Pavankumar | Image Processing + ML | `feature/image-processing` | `image-processing/`, `models/` |
| Sanjeev | Video Processing | `feature/video-processing` | `video-processing/` |

`main` is the stable integration branch. Nobody should develop directly on `main`.

---

# 2. Overall System Flow

```text
                         USER
                           |
                           v
                    +-------------+
                    |  FRONTEND   |
                    |   Yashas    |
                    +------+------+
                           |
                       REST API
                           |
                           v
                    +-------------+
                    |   BACKEND   |
                    |    Vikas    |
                    +------+------+
                           |
                +----------+----------+
                |                     |
                v                     v
        +---------------+     +---------------+
        | IMAGE PROCESS |     | VIDEO PROCESS |
        |    Pavankumar |     |    Sanjeev    |
        +-------+-------+     +-------+-------+
                |                     |
                +----------+----------+
                           |
                           v
                     ML INFERENCE
                           |
                           v
                       RESULT
                           |
                           v
                        BACKEND
                           |
                           v
                       FRONTEND
                           |
                           v
                          USER
```

The exact internal implementation may change, but the module boundaries should remain clear.

---

# 3. The Core Rule

Every module must clearly define:

```text
INPUT → PROCESSING → OUTPUT
```

A developer should not need to know the internal implementation of another person's module.

For example, Vikas should be able to call the image-processing module without knowing which face detector, preprocessing technique, or ML architecture Pavankumar used.

---

# 4. YASHAS — FRONTEND

## Responsibility

Build the user-facing DeepGuard application.

### Owns

```text
frontend/
```

### Branch

```text
feature/frontend
```

## Frontend should provide

- Image upload UI
- Video upload UI
- File validation
- Loading/progress state
- Error state
- Result display
- Confidence display
- API service/client layer
- Clean and responsive UI

## Frontend expects

The backend to provide HTTP/REST endpoints.

Initially, Yashas can use mock responses so frontend development does not depend on the backend being finished.

Example expected image response:

```json
{
  "status": "success",
  "media_type": "image",
  "prediction": "FAKE",
  "confidence": 0.94
}
```

Example expected video response:

```json
{
  "status": "success",
  "media_type": "video",
  "prediction": "FAKE",
  "confidence": 0.91,
  "frames_analyzed": 30
}
```

These are initial contracts and may be extended after integration.

## Frontend must return

A working frontend that can:

```text
Select media
    ↓
Upload media
    ↓
Call backend
    ↓
Handle loading/error
    ↓
Receive result
    ↓
Display result
```

## Yashas does NOT own

- ML model
- Image processing algorithms
- Video frame extraction
- Backend business logic

---

# 5. VIKAS — BACKEND / API

## Responsibility

Act as the central API and orchestration layer.

### Owns

```text
backend/
```

### Branch

```text
feature/backend
```

## Backend expects

From the frontend:

```text
Image file
or
Video file
```

Example:

```http
POST /api/detect/image
Content-Type: multipart/form-data
```

and:

```http
POST /api/detect/video
Content-Type: multipart/form-data
```

## Backend should do

```text
Receive request
    ↓
Validate input
    ↓
Identify media type
    ↓
Call correct processing module
    ↓
Receive standardized result
    ↓
Return API response
```

## Backend expects from Pavankumar

A callable image-detection interface.

Conceptually:

```python
result = predict_image(image_path)
```

Expected result:

```json
{
  "prediction": "FAKE",
  "confidence": 0.94
}
```

## Backend expects from Sanjeev

A callable video-processing interface.

Conceptually:

```python
result = process_video(video_path)
```

Expected result:

```json
{
  "prediction": "FAKE",
  "confidence": 0.91,
  "frames_analyzed": 30
}
```

The exact Python function names can be agreed during integration.

## Backend must return

A stable HTTP API response for the frontend.

It should also handle:

- invalid files
- unsupported formats
- processing failures
- timeouts/errors
- configuration
- logging

## Vikas does NOT own

- Face detection implementation
- Model training
- Video frame algorithms
- Frontend UI

---

# 6. PAVANKUMAR — IMAGE PROCESSING + ML

## Responsibility

Build the image deepfake detection pipeline and the ML inference layer.

### Owns

```text
image-processing/
models/
```

### Branch

```text
feature/image-processing
```

## Image-processing input

The module should eventually accept:

```text
Image file/path
```

Conceptual interface:

```python
result = predict_image(image_path)
```

## Internal pipeline

The implementation may look like:

```text
Image
  ↓
Image loading
  ↓
Face detection
  ↓
Face extraction/cropping
  ↓
Resize / normalization
  ↓
Model inference
  ↓
Prediction
```

## Image-processing output

At minimum:

```json
{
  "prediction": "REAL",
  "confidence": 0.87
}
```

Additional metadata can be added if genuinely useful, for example:

```json
{
  "prediction": "FAKE",
  "confidence": 0.94,
  "face_detected": true,
  "faces_analyzed": 1
}
```

Do not add fields just because they are possible. Keep the contract simple.

## ML responsibility

Pavankumar should define and document:

- model architecture
- model input dimensions
- preprocessing
- model loading
- inference
- output interpretation
- confidence calculation
- model-weight/setup requirements

## Important

Large model files should NOT automatically be committed to Git.

Use the approach agreed by the team (for example Git LFS or external model storage) if the weights are too large.

## Pavankumar must return

A reusable image inference component:

```text
image → prediction + confidence
```

with:

- working inference
- documented setup
- tests
- reproducible example
- clear input/output contract

---

# 7. SANJEEV — VIDEO PROCESSING

## Responsibility

Build the video-specific processing pipeline.

### Owns

```text
video-processing/
```

### Branch

```text
feature/video-processing
```

## Video input

The module accepts:

```text
Video file/path
```

Conceptual interface:

```python
result = process_video(video_path)
```

## Internal pipeline

Potential flow:

```text
Video
  ↓
Open video
  ↓
Extract frames
  ↓
Sample frames
  ↓
Detect/extract faces
  ↓
Preprocess
  ↓
Run ML inference
  ↓
Collect frame predictions
  ↓
Aggregate predictions
  ↓
Final video result
```

## Video output

At minimum:

```json
{
  "prediction": "FAKE",
  "confidence": 0.91,
  "frames_analyzed": 30
}
```

Additional useful metadata can be added later.

## Sanjeev expects

A clearly documented ML inference interface from Pavankumar.

For example:

```text
face/frame → model prediction + confidence
```

Sanjeev should NOT create an unrelated model pipeline if the project intends to use the same model/inference layer.

## Sanjeev must return

A reusable video-processing component:

```text
video → prediction + confidence + processing metadata
```

with:

- frame extraction
- frame sampling
- preprocessing
- inference integration
- prediction aggregation
- tests
- setup documentation

---

# 8. PAVANKUMAR ↔ SANJEEV: ML CONTRACT

This is the most important cross-team agreement.

Before deep implementation, both members must agree on:

### Model input

For example:

```text
224 × 224 RGB face image
```

### Model output

For example:

```text
fake_probability
```

or:

```json
{
  "prediction": "FAKE",
  "confidence": 0.94
}
```

The actual dimensions and output format must be determined by the selected model.

The goal is:

```text
IMAGE PROCESSING ──┐
                   ├──> COMMON ML INFERENCE
VIDEO PROCESSING ──┘
```

---

# 9. MODULE CONTRACTS

## Frontend → Backend

```text
Input:
Image or video upload

Transport:
HTTP/REST

Output:
Standardized JSON response
```

## Backend → Image Processing

```text
Input:
Image path/file

Output:
Prediction + confidence
```

## Backend → Video Processing

```text
Input:
Video path/file

Output:
Prediction + confidence + video metadata
```

## Video Processing → ML

```text
Input:
Preprocessed frame/face

Output:
Model prediction + confidence
```

## Image Processing → ML

```text
Input:
Preprocessed face/image

Output:
Model prediction + confidence
```

---

# 10. Development Strategy

Everyone can work independently.

## Phase 1 — Independent development

### Yashas

Use mock API responses.

### Vikas

Use mock image/video processing functions.

### Pavankumar

Use test images and build actual image inference.

### Sanjeev

Use test videos and build video processing.

Nobody needs to wait for another member.

---

# 11. Phase 2 — Integration

Integration order:

```text
1. Frontend ↔ Backend

2. Backend ↔ Image Processing

3. Backend ↔ Video Processing

4. Image/Video ↔ ML

5. Full system test
```

Final flow:

```text
User
 ↓
Frontend
 ↓
Backend
 ↓
Image / Video Processing
 ↓
ML
 ↓
Prediction
 ↓
Backend
 ↓
Frontend
 ↓
User
```

---

# 12. Definition of Done

A module is NOT complete just because the code has been pushed.

Each member must provide:

### Required

- Working implementation
- `README.md`
- Setup instructions
- Input/output documentation
- Example usage
- Tests
- No secrets/API keys
- No unnecessary dependencies
- PR to `main`

---

# 13. Git Workflow

## Before starting

```bash
git fetch origin
git checkout feature/<your-branch>
git merge origin/main
```

## During development

```bash
git add .
git commit -m "Describe the change"
```

## Push

```bash
git push origin feature/<your-branch>
```

## Integration

```text
Feature branch
      ↓
     PR
      ↓
Code review
      ↓
Testing
      ↓
Merge
      ↓
main
```

Nobody directly develops on `main`.

---

# 14. What Each Person Must Deliver

| Member | Must deliver | Next person depends on |
|---|---|---|
| **Yashas** | Working UI + API client | Backend API |
| **Vikas** | REST API + orchestration | Image/Video modules |
| **Pavankumar** | Image preprocessing + ML inference | Backend + Video module |
| **Sanjeev** | Video processing + frame pipeline | Backend + ML inference |

---

# 15. Current Phase

Since the project is starting from scratch except for the existing backend:

## DO NOW

```text
Yashas     → Start frontend
Vikas      → Clean/integrate existing backend
Pavankumar → Start image-processing + ML research
Sanjeev    → Start video-processing
```

## DO NOT DO YET

- Don't merge everything into one person's branch.
- Don't duplicate ML models.
- Don't hard-code assumptions about another module.
- Don't put frontend logic in backend.
- Don't put video processing inside backend.
- Don't put API logic inside ML code.
- Don't commit secrets.
- Don't directly push to `main`.

---

# 16. Integration Principle

Every member should be able to say:

> **"This is what my module accepts, this is what it returns, and this is how another module calls it."**

If that is true, the four independent pieces can be integrated without major rewrites.
