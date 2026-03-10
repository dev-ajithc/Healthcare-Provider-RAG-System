import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.rag.pipeline import process_query
from app.schemas import ErrorDetail, ErrorResponse, QueryRequest, QueryResponse

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def query_endpoint(
    payload: QueryRequest,
    request: Request,
) -> QueryResponse:
    if settings.query_killswitch:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error=ErrorDetail(
                    code="SERVICE_UNAVAILABLE",
                    message=(
                        "The query service is temporarily disabled."
                    ),
                    request_id=getattr(
                        request.state, "request_id", None
                    ),
                    retry_after=60,
                )
            ).model_dump(),
        )

    try:
        request_id = getattr(
            request.state, "request_id", None
        )
        response = await process_query(payload, request_id)
        return response
    except RuntimeError as exc:
        if "QUERY_KILLSWITCH" in str(exc):
            raise HTTPException(status_code=503, detail=str(exc))
        logger.error(
            "query_runtime_error",
            error=str(exc),
            request_id=getattr(
                request.state, "request_id", None
            ),
        )
        raise HTTPException(
            status_code=503,
            detail="Query processing failed. Please try again.",
        )
    except Exception as exc:
        logger.error(
            "query_unexpected_error",
            error=str(exc),
            request_id=getattr(
                request.state, "request_id", None
            ),
        )
        raise HTTPException(
            status_code=503,
            detail="An unexpected error occurred.",
        )
