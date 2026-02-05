"""AI Models API routes."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_models():
    """Get model registry."""
    return {"models": []}


@router.get("/active")
async def get_active_model():
    """Get currently active model."""
    return {"model": None}


@router.post("/{version}/activate")
async def activate_model(version: str):
    """Activate a model version."""
    return {"success": True, "version": version}


@router.get("/health")
async def get_model_health():
    """Get model health/drift indicators."""
    return {"drift_detected": False, "metrics": {}}
