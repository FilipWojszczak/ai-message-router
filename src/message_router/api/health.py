from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


@router.get("/health")
async def health() -> HealthResponse:
    """Liveness only: no I/O, no dependency checks."""
    return HealthResponse()
