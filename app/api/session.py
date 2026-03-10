import uuid

import structlog
from fastapi import APIRouter, HTTPException

from app.rag.pipeline import clear_session, get_session_history
from app.schemas import SessionMessage, SessionResponse

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get(
    "/session/{session_id}",
    response_model=SessionResponse,
)
async def get_session_endpoint(
    session_id: uuid.UUID,
) -> SessionResponse:
    messages = await get_session_history(str(session_id))
    return SessionResponse(
        session_id=session_id,
        messages=[
            SessionMessage(
                role=m.get("role", "user"),
                content=m.get("content", ""),
            )
            for m in messages
        ],
    )


@router.delete("/session/{session_id}", status_code=204)
async def delete_session_endpoint(
    session_id: uuid.UUID,
) -> None:
    await clear_session(str(session_id))
