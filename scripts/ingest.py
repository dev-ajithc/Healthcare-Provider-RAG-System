"""Chunk, embed, and upsert synthetic provider data into pgvector."""

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
import tiktoken
from sqlalchemy import text

sys.path.insert(
    0, str(Path(__file__).parent.parent)
)

from app.core.config import settings
from app.db.session import AsyncSessionLocal, engine
from app.rag.embeddings import embed_batch

logger = structlog.get_logger(__name__)

CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
EMBED_BATCH_SIZE = 100
DB_BATCH_SIZE = 500

_tokeniser = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_tokeniser.encode(text))


def _chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    tokens = _tokeniser.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = _tokeniser.decode(chunk_tokens)
        if chunk_text.strip():
            chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks


def _build_chunk_content(
    provider: Dict[str, Any],
    chunk_text: str,
) -> str:
    specialties = ", ".join(provider.get("specialties", []))
    insurances = ", ".join(provider.get("insurances", []))
    return (
        f"NPI: {provider['npi']} | "
        f"Name: {provider['name']} | "
        f"State: {provider['state']} | "
        f"Specialty: {specialties} | "
        f"Insurance: {insurances} | "
        f"Rating: {provider.get('rating', 'N/A')} | "
        f"Accepting: {provider.get('accepting_new_patients', False)}\n"
        f"{chunk_text}"
    )


def _validate_embedding(emb: List[float]) -> bool:
    arr = np.array(emb, dtype=np.float32)
    return bool(np.all(np.isfinite(arr)))


async def _ensure_pgvector_extension(conn: Any) -> None:
    await conn.execute(
        text("CREATE EXTENSION IF NOT EXISTS vector")
    )


async def _upsert_provider(
    conn: Any,
    provider: Dict[str, Any],
) -> uuid.UUID:
    provider_id = uuid.uuid4()
    await conn.execute(
        text("""
            INSERT INTO providers (
                id, npi, name, gender, specialties, state,
                city, address, lat, long, insurances,
                rating, accepting_new_patients, bio
            ) VALUES (
                :id, :npi, :name, :gender, :specialties, :state,
                :city, :address, :lat, :long, :insurances,
                :rating, :accepting_new_patients, :bio
            )
            ON CONFLICT (npi) DO UPDATE SET
                name = EXCLUDED.name,
                gender = EXCLUDED.gender,
                specialties = EXCLUDED.specialties,
                state = EXCLUDED.state,
                city = EXCLUDED.city,
                address = EXCLUDED.address,
                lat = EXCLUDED.lat,
                long = EXCLUDED.long,
                insurances = EXCLUDED.insurances,
                rating = EXCLUDED.rating,
                accepting_new_patients = EXCLUDED.accepting_new_patients,
                bio = EXCLUDED.bio
            RETURNING id
        """),
        {
            "id": str(provider_id),
            "npi": provider["npi"],
            "name": provider["name"],
            "gender": provider["gender"],
            "specialties": provider["specialties"],
            "state": provider["state"],
            "city": provider["city"],
            "address": provider["address"],
            "lat": float(provider["lat"]),
            "long": float(provider["long"]),
            "insurances": provider["insurances"],
            "rating": float(provider["rating"])
            if provider.get("rating")
            else None,
            "accepting_new_patients": provider.get(
                "accepting_new_patients", True
            ),
            "bio": provider["bio"],
        },
    )
    result = await conn.execute(
        text("SELECT id FROM providers WHERE npi = :npi"),
        {"npi": provider["npi"]},
    )
    row = result.first()
    return uuid.UUID(str(row[0])) if row else provider_id


async def _upsert_embeddings_batch(
    conn: Any,
    batch: List[Tuple[uuid.UUID, int, str, List[float], int]],
) -> None:
    for provider_id, chunk_idx, content, embedding, token_count in batch:
        emb_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
        await conn.execute(
            text("""
                INSERT INTO embeddings (
                    id, provider_id, chunk_index, content,
                    embedding, token_count, model
                ) VALUES (
                    :id, :provider_id, :chunk_index, :content,
                    CAST(:embedding AS vector), :token_count, :model
                )
                ON CONFLICT (provider_id, chunk_index) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    token_count = EXCLUDED.token_count,
                    model = EXCLUDED.model
            """),
            {
                "id": str(uuid.uuid4()),
                "provider_id": str(provider_id),
                "chunk_index": chunk_idx,
                "content": content,
                "embedding": emb_str,
                "token_count": token_count,
                "model": settings.embedding_model,
            },
        )


async def _create_job(conn: Any, total: int) -> uuid.UUID:
    job_id = uuid.uuid4()
    await conn.execute(
        text("""
            INSERT INTO ingest_jobs
                (id, status, total, processed, errors, started_at)
            VALUES (:id, 'running', :total, 0, 0, NOW())
        """),
        {"id": str(job_id), "total": total},
    )
    await conn.commit()
    return job_id


async def _update_job(
    conn: Any,
    job_id: uuid.UUID,
    processed: int,
    errors: int,
    status: str,
    error_msg: Optional[str] = None,
) -> None:
    finished = status in ("done", "failed")
    finished_sql = ", finished_at = NOW()" if finished else ""
    await conn.execute(
        text(f"""
            UPDATE ingest_jobs
            SET status = :status,
                processed = :processed,
                errors = :errors,
                error_msg = :error_msg
                {finished_sql}
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
    await conn.commit()


async def run_ingest(
    providers: List[Dict[str, Any]],
    verbose: bool = True,
) -> None:
    total = len(providers)
    processed = 0
    errors = 0

    async with engine.begin() as conn:
        await _ensure_pgvector_extension(conn)
        await conn.commit()

    async with AsyncSessionLocal() as db_session:
        job_id = await _create_job(db_session, total)

    if verbose:
        print(f"Starting ingestion of {total} providers...")
        print(f"Job ID: {job_id}")

    for batch_start in range(0, total, DB_BATCH_SIZE):
        batch_providers = providers[
            batch_start: batch_start + DB_BATCH_SIZE
        ]

        all_chunks: List[
            Tuple[Dict[str, Any], int, str]
        ] = []
        for provider in batch_providers:
            raw_chunks = _chunk_text(provider["bio"])
            for idx, chunk in enumerate(raw_chunks):
                content = _build_chunk_content(
                    provider, chunk
                )
                all_chunks.append((provider, idx, content))

        chunk_texts = [c[2] for c in all_chunks]
        embeddings: List[List[float]] = []

        for emb_start in range(
            0, len(chunk_texts), EMBED_BATCH_SIZE
        ):
            emb_batch = chunk_texts[
                emb_start: emb_start + EMBED_BATCH_SIZE
            ]
            try:
                batch_embs = await embed_batch(emb_batch)
                embeddings.extend(batch_embs)
            except Exception as e:
                logger.error(
                    "embed_batch_failed",
                    start=emb_start,
                    error=str(e),
                )
                embeddings.extend(
                    [[0.0] * settings.embedding_dimensions]
                    * len(emb_batch)
                )
                errors += len(emb_batch)

        async with engine.begin() as conn:
            for i, provider in enumerate(batch_providers):
                try:
                    provider_id = await _upsert_provider(
                        conn, provider
                    )

                    emb_batch_for_provider = []
                    for flat_idx, (
                        prov, chunk_local_idx, content
                    ) in enumerate(all_chunks):
                        if prov["npi"] != provider["npi"]:
                            continue
                        if flat_idx >= len(embeddings):
                            continue
                        emb = embeddings[flat_idx]
                        if not _validate_embedding(emb):
                            errors += 1
                            continue
                        token_count = _count_tokens(content)
                        emb_batch_for_provider.append(
                            (
                                provider_id,
                                chunk_local_idx,
                                content,
                                emb,
                                token_count,
                            )
                        )

                    if emb_batch_for_provider:
                        await _upsert_embeddings_batch(
                            conn, emb_batch_for_provider
                        )

                    processed += 1

                except Exception as e:
                    errors += 1
                    logger.error(
                        "provider_ingest_failed",
                        npi=provider.get("npi"),
                        error=str(e),
                    )

            await conn.commit()

        async with AsyncSessionLocal() as db_session:
            await _update_job(
                db_session,
                job_id,
                processed,
                errors,
                "running",
            )

        if verbose:
            pct = processed / total * 100
            print(
                f"Progress: {processed}/{total} "
                f"({pct:.1f}%) | Errors: {errors}"
            )

    async with AsyncSessionLocal() as db_session:
        status = "done" if errors == 0 else "done_with_errors"
        await _update_job(
            db_session, job_id, processed, errors, status
        )

    if verbose:
        print(
            f"\nIngestion complete: "
            f"{processed} providers, {errors} errors."
        )


async def main() -> None:
    data_file = (
        Path(__file__).parent.parent / "data" / "providers.json"
    )
    if not data_file.exists():
        print(
            "providers.json not found. "
            "Run scripts/generate_data.py first."
        )
        sys.exit(1)

    with open(data_file) as f:
        providers = json.load(f)

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if limit:
        providers = providers[:limit]

    await run_ingest(providers, verbose=True)


if __name__ == "__main__":
    asyncio.run(main())
