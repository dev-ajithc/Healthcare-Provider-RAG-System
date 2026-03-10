import uuid
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


async def get_provider_count(db: AsyncSession) -> int:
    result = await db.execute(text("SELECT COUNT(*) FROM providers"))
    return result.scalar_one()


async def hybrid_retrieve(
    db: AsyncSession,
    query_embedding: List[float],
    query_text: str,
    top_k: int = 20,
    similarity_cutoff: float = 0.5,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    filter_clauses = []
    filter_params: Dict[str, Any] = {
        "embedding": query_embedding,
        "query_text": query_text,
        "cutoff": similarity_cutoff,
        "top_k": top_k,
    }

    if filters:
        if filters.get("state"):
            filter_clauses.append("p.state = :state")
            filter_params["state"] = filters["state"].upper()
        if filters.get("insurance"):
            filter_clauses.append(
                ":insurance = ANY(p.insurances)"
            )
            filter_params["insurance"] = filters["insurance"]
        if filters.get("accepting_new_patients"):
            filter_clauses.append(
                "p.accepting_new_patients = TRUE"
            )

    extra_filters = (
        " AND " + " AND ".join(filter_clauses)
        if filter_clauses
        else ""
    )

    dense_sql = text(f"""
        SELECT
            e.id AS embedding_id,
            e.provider_id,
            e.content,
            e.chunk_index,
            (e.embedding <=> CAST(:embedding AS vector)) AS distance,
            p.npi,
            p.name,
            p.specialties,
            p.state,
            p.city,
            p.insurances,
            p.rating,
            p.accepting_new_patients,
            p.lat,
            p.long
        FROM embeddings e
        JOIN providers p ON p.id = e.provider_id
        WHERE (e.embedding <=> CAST(:embedding AS vector)) < :cutoff
        {extra_filters}
        ORDER BY distance
        LIMIT :top_k
    """)

    bm25_sql = text(f"""
        SELECT
            NULL AS embedding_id,
            p.id AS provider_id,
            p.bio AS content,
            0 AS chunk_index,
            NULL AS distance,
            p.npi,
            p.name,
            p.specialties,
            p.state,
            p.city,
            p.insurances,
            p.rating,
            p.accepting_new_patients,
            p.lat,
            p.long,
            ts_rank(
                p.bio_tsv,
                plainto_tsquery('english', :query_text)
            ) AS bm25_rank
        FROM providers p
        WHERE p.bio_tsv @@ plainto_tsquery('english', :query_text)
        {extra_filters}
        ORDER BY bm25_rank DESC
        LIMIT :top_k
    """)

    try:
        dense_result = await db.execute(dense_sql, filter_params)
        dense_rows = dense_result.mappings().all()
    except Exception as e:
        logger.warning("dense_retrieval_failed", error=str(e))
        dense_rows = []

    try:
        bm25_result = await db.execute(bm25_sql, filter_params)
        bm25_rows = bm25_result.mappings().all()
    except Exception as e:
        logger.warning("bm25_retrieval_failed", error=str(e))
        bm25_rows = []

    return _rrf_fusion(dense_rows, bm25_rows, top_k)


def _rrf_fusion(
    dense_rows: List[Any],
    bm25_rows: List[Any],
    top_k: int,
    k: int = 60,
) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    rows_by_key: Dict[str, Dict[str, Any]] = {}

    for rank, row in enumerate(dense_rows, start=1):
        key = f"{row['provider_id']}_{row['chunk_index']}"
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        rows_by_key[key] = dict(row)

    for rank, row in enumerate(bm25_rows, start=1):
        key = f"{row['provider_id']}_0"
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in rows_by_key:
            rows_by_key[key] = dict(row)

    sorted_keys = sorted(
        scores.keys(), key=lambda x: scores[x], reverse=True
    )
    results = []
    for key in sorted_keys[:top_k]:
        row = rows_by_key[key]
        row["rrf_score"] = scores[key]
        results.append(row)

    return results


async def insert_audit_log(
    db: AsyncSession,
    session_id: uuid.UUID,
    query_hash: str,
    latency_ms: int,
    token_in: int,
    token_out: int,
    cache_hit: bool,
) -> None:
    await db.execute(
        text("""
            INSERT INTO audit_logs
                (session_id, query_hash, latency_ms,
                 token_in, token_out, cache_hit)
            VALUES
                (:session_id, :query_hash, :latency_ms,
                 :token_in, :token_out, :cache_hit)
        """),
        {
            "session_id": str(session_id),
            "query_hash": query_hash,
            "latency_ms": latency_ms,
            "token_in": token_in,
            "token_out": token_out,
            "cache_hit": cache_hit,
        },
    )
    await db.commit()


async def get_latest_ingest_job(
    db: AsyncSession,
) -> Optional[Dict[str, Any]]:
    result = await db.execute(
        text("""
            SELECT id, status, total, processed, errors,
                   started_at, finished_at, error_msg
            FROM ingest_jobs
            ORDER BY started_at DESC NULLS LAST
            LIMIT 1
        """)
    )
    row = result.mappings().first()
    return dict(row) if row else None


async def create_ingest_job(
    db: AsyncSession,
    total: int,
) -> uuid.UUID:
    job_id = uuid.uuid4()
    await db.execute(
        text("""
            INSERT INTO ingest_jobs
                (id, status, total, processed, errors, started_at)
            VALUES
                (:id, 'running', :total, 0, 0, NOW())
        """),
        {"id": str(job_id), "total": total},
    )
    await db.commit()
    return job_id


async def update_ingest_job(
    db: AsyncSession,
    job_id: uuid.UUID,
    processed: int,
    errors: int,
    status: str,
    error_msg: Optional[str] = None,
) -> None:
    finished_clause = (
        ", finished_at = NOW()"
        if status in ("done", "failed")
        else ""
    )
    await db.execute(
        text(f"""
            UPDATE ingest_jobs
            SET status = :status,
                processed = :processed,
                errors = :errors,
                error_msg = :error_msg
                {finished_clause}
            WHERE id = :job_id
        """),
        {
            "status": status,
            "processed": processed,
            "errors": errors,
            "error_msg": error_msg,
            "job_id": str(job_id),
        },
    )
    await db.commit()
