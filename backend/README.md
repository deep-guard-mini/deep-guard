# DeepGuard Backend

REST API and orchestration layer for the DeepGuard deepfake detection system.

> **Phase 1 — Demo mode.**  
> ML inference is mocked. Results are randomly generated so the full system
> can be developed and tested end-to-end before the ML modules are integrated.

---

## Stack

| Component | Technology |
|---|---|
| Framework | FastAPI |
| Server | Uvicorn |
| Language | Python 3.10+ |
| File uploads | python-multipart |
| Testing | pytest + httpx |

---

## Project structure

```
backend/
├── app.py                  ← FastAPI app, CORS, router registration
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── routers/
│   ├── health.py           ← GET /api/health
│   └── detect.py           ← POST /api/detect/image, POST /api/detect/video
│
├── services/
│   ├── image_service.py    ← Phase 1 mock; replace with real ML in Phase 3
│   └── video_service.py    ← Phase 1 mock; replace with real ML in Phase 3
│
└── utils/
    ├── file_validation.py  ← Magic-byte format detection, size limit
    └── response_builder.py ← Maps service output → frontend JSON contract
```

Tests live in the top-level `tests/` directory (sibling of `backend/`).

---

## Requirements

- Python **3.10** or newer
- pip

---

## Installation

Run all commands from the **repository root** (`deep-guard/`).

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

---

## Running the development server

From the **repository root**:

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

The API is now available at:

| URL | Purpose |
|---|---|
| `http://localhost:8000/api/health` | Health check |
| `http://localhost:8000/api/detect/image` | Image analysis endpoint |
| `http://localhost:8000/api/detect/video` | Video analysis endpoint |
| `http://localhost:8000/api/docs` | Interactive Swagger UI |
| `http://localhost:8000/api/redoc` | ReDoc documentation |

---

## API reference

### `GET /api/health`

Returns `200 OK` when the service is running.

```json
{
  "status": "ok",
  "service": "DeepGuard Backend",
  "version": "1.0.0-phase1"
}
```

---

### `POST /api/detect/image`

Analyse an image for deepfake indicators.

**Request**

```
Content-Type: multipart/form-data
Field name:   file
Accepted:     JPEG, PNG, WebP
Max size:     100 MB
```

**Response `200 OK`**

```json
{
  "status":            "success",
  "prediction":        "FAKE",
  "confidence":        94.2,
  "assessment":        "LIKELY MANIPULATED",
  "assessmentTone":    "danger",
  "mediaType":         "image",
  "processingTime":    0.83,
  "facesDetected":     2,
  "framesAnalyzed":    1,
  "artifactsDetected": 7,
  "indicators": {
    "faceConsistency":      82,
    "textureConsistency":   79,
    "lightingConsistency":  71,
    "compressionArtifacts": 88,
    "edgeArtifacts":        91
  },
  "note":     "[DEMO — No ML model active] ...",
  "demoMode": true
}
```

**Error responses**

| Status | `error` code | Cause |
|---|---|---|
| 400 | `MISSING_FILE` | No file in the request |
| 413 | `FILE_TOO_LARGE` | File exceeds 100 MB |
| 415 | `UNSUPPORTED_FORMAT` | Not a JPEG, PNG or WebP |
| 500 | `PROCESSING_FAILED` | Internal service error |

---

### `POST /api/detect/video`

Analyse a video for deepfake indicators.

**Request**

```
Content-Type: multipart/form-data
Field name:   file
Accepted:     MP4, MOV, AVI
Max size:     100 MB
```

**Response** — same shape as image, with `"mediaType": "video"` and
`"framesAnalyzed"` > 1.

---

## Running the tests

From the **repository root**:

```bash
pytest
```

With verbose output:

```bash
pytest -v
```

Single test class:

```bash
pytest tests/test_backend.py::TestHealth -v
```

---

## Integration — connecting the frontend

When the backend is running on port 8000, update the Vite proxy config in
`frontend/src/vite.config.js` (coordinate with Yashas):

```js
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

Then update `frontend/src/services/api.js` to make real `fetch` calls instead
of returning mock data.

---

## Integrating real ML modules (Phase 3)

| Step | What to do |
|---|---|
| Image | Replace `backend/services/image_service.py::process_image()` with a call to Pavankumar's `predict_image()`. See the module docstring for the exact key contract. |
| Video | Replace `backend/services/video_service.py::process_video()` with a call to Sanjeev's `process_video()`. See the module docstring for the exact key contract. |

Only the service files change. The routers and response builder are untouched.

---

## Known limitations (Phase 1)

- **Mock results only.** No real deepfake detection occurs.
- **100 MB upload read into memory.** For production, streaming disk writes
  should be used for large files.
- **Sync service calls inside async endpoints.** The mock is fast enough that
  this does not block the event loop. In Phase 3, consider wrapping
  CPU-intensive inference in `asyncio.get_event_loop().run_in_executor()`.
- **CORS origin is hardcoded** to `http://localhost:5173`. Must be updated
  before any deployment.
- **No authentication or rate limiting.** Not needed for Phase 1.
