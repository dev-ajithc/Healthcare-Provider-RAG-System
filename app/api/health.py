import time

import structlog
from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from app.core.config import settings
from app.db.session import engine
from app.rag.cache import check_redis
from app.rag.embeddings import check_embed_api
from app.schemas import HealthResponse

logger = structlog.get_logger(__name__)
router = APIRouter()

_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    db_status = "ok"
    redis_status = "ok"
    embed_status = "ok"
    overall = "ok"

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        db_status = "down"
        overall = "degraded"
        logger.error("db_health_check_failed", error=str(e))

    redis_ok = await check_redis()
    if not redis_ok:
        redis_status = "down"
        overall = "degraded"
        logger.warning("redis_health_check_failed")

    embed_ok = await check_embed_api()
    if not embed_ok:
        embed_status = "down"
        overall = "degraded"
        logger.warning("embed_api_health_check_failed")

    if db_status == "down":
        overall = "down"

    return HealthResponse(
        status=overall,
        db=db_status,
        redis=redis_status,
        embed_api=embed_status,
        version=settings.app_version,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@router.get("/metrics")
async def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
