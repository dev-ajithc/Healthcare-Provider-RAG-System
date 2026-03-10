import structlog
from fastapi import APIRouter, Header, HTTPException

from app.core.config import settings
from app.core.security import validate_admin_key
from app.db.queries import get_latest_ingest_job
from app.db.session import AsyncSessionLocal
from app.rag.cache import flush_semantic_cache
from app.schemas import IngestStatusResponse

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get(
    "/ingest/status",
    response_model=IngestStatusResponse,
)
async def ingest_status() -> IngestStatusResponse:
    async with AsyncSessionLocal() as db:
        job = await get_latest_ingest_job(db)

    if not job:
        return IngestStatusResponse(
            job_id=None,
            status="no_jobs",
            total=None,
            processed=0,
            errors=0,
            started_at=None,
            finished_at=None,
            error_msg=None,
        )

    return IngestStatusResponse(
        job_id=str(job["id"]),
        status=str(job["status"]),
        total=job.get("total"),
        processed=int(job.get("processed", 0)),
        errors=int(job.get("errors", 0)),
        started_at=(
            str(job["started_at"])
            if job.get("started_at")
            else None
        ),
        finished_at=(
            str(job["finished_at"])
            if job.get("finished_at")
            else None
        ),
        error_msg=job.get("error_msg"),
    )


@router.post("/ingest/trigger", status_code=202)
async def trigger_ingest(
    x_admin_key: str = Header(..., alias="X-Admin-Key"),
) -> dict:
    if not validate_admin_key(
        x_admin_key, settings.admin_key
    ):
        raise HTTPException(
            status_code=403, detail="Invalid admin key"
        )

    flushed = await flush_semantic_cache()
    logger.info(
        "ingest_triggered",
        cache_keys_flushed=flushed,
    )

    return {
        "message": (
            "Re-ingestion triggered. "
            "Run scripts/ingest.py to populate data. "
            f"Cache flushed ({flushed} entries)."
        )
    }
