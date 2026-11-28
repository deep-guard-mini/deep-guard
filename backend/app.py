"""
DeepGuard Backend — FastAPI application entry point.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Quick start (from the repository root):
    uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

Interactive API docs:
    http://localhost:8000/api/docs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import detect, health

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DeepGuard API",
    description=(
        "AI-based deepfake detection API for image and video analysis.\n\n"
        "Image analysis uses a trained ResNet50 model; video analysis uses a "
        "trained FakeSTormer C23 spatio-temporal ViT model."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# Default development origins for local Vite / frontend dev servers.
DEFAULT_DEV_ORIGINS: list[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def get_allowed_origins() -> list[str]:
    """
    Parse allowed CORS origins from the ALLOWED_ORIGINS environment variable.

    Supports comma-separated URLs, strips whitespace, and removes empty items.
    Always includes local development origins so local testing continues working.
    Avoids accidentally allowing unrestricted wildcard '*' origins in production.
    """
    origins = list(DEFAULT_DEV_ORIGINS)
    raw_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
    if raw_origins:
        for entry in raw_origins.split(","):
            cleaned = entry.strip()
            if cleaned and cleaned not in origins:
                origins.append(cleaned)
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(health.router)
app.include_router(detect.router)


# ---------------------------------------------------------------------------
# Direct execution / platform entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, reload=False)
