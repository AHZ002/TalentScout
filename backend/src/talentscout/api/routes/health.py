'''provide an endpoint that tells you whether the backend is running.'''

from typing import Final

from fastapi import APIRouter #lets you group related API endpoints together.
from pydantic import BaseModel

router = APIRouter(tags=["health"])

HEALTH_STATUS: Final[str] = "ok"


class HealthResponse(BaseModel):
    status: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status=HEALTH_STATUS)
