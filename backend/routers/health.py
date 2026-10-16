"""Health check router — GET /api/health"""

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", summary="Service health check")
async def health() -> dict:
    """
    Returns 200 OK when the backend service is running.

    The frontend or any monitoring tool can poll this endpoint to verify
    that the backend is reachable before submitting media for analysis.
    """
    return {
        "status":  "ok",
        "service": "DeepGuard Backend",
        "version": "1.0.0-phase1",
    }
