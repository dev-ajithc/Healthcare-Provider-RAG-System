import json
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import redis.asyncio as aioredis
import structlog

from app.core.config import settings
from app.core.security import hash_query

logger = structlog.get_logger(__name__)

_redis_client: aioredis.Redis | None = None
CACHE_KEY_PREFIX = "sem_cache:"
EMBEDDING_KEY_PREFIX = "sem_emb:"


def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.redis_url,
            decode_responses=False,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
    return _redis_client


def _encode_embedding(embedding: List[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def _decode_embedding(data: bytes) -> List[float]:
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


def _cosine_similarity(
    a: List[float], b: List[float]
) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


async def semantic_cache_lookup(
    query_embedding: List[float],
    threshold: float = None,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    if threshold is None:
        threshold = settings.cache_similarity_threshold
    try:
        redis = get_redis()
        keys = await redis.keys(f"{EMBEDDING_KEY_PREFIX}*")
        best_sim = 0.0
        best_hash: Optional[str] = None

        for key in keys:
            raw = await redis.get(key)
            if raw is None:
                continue
            cached_emb = _decode_embedding(raw)
            sim = _cosine_similarity(query_embedding, cached_emb)
            if sim > best_sim:
                best_sim = sim
                best_hash = key.decode().replace(
                    EMBEDDING_KEY_PREFIX, ""
                )

        if best_sim >= threshold and best_hash:
            cached_resp = await redis.get(
                f"{CACHE_KEY_PREFIX}{best_hash}"
            )
            if cached_resp:
                logger.info(
                    "cache_hit",
                    similarity=round(best_sim, 4),
                )
                return json.loads(cached_resp), True

    except Exception as e:
        logger.warning("cache_lookup_failed", error=str(e))

    return None, False


async def semantic_cache_store(
    query: str,
    query_embedding: List[float],
    response: Dict[str, Any],
    ttl: int = None,
) -> None:
    if ttl is None:
        ttl = settings.cache_ttl_seconds
    try:
        redis = get_redis()
        q_hash = hash_query(query)
        emb_bytes = _encode_embedding(query_embedding)
        await redis.setex(
            f"{EMBEDDING_KEY_PREFIX}{q_hash}",
            ttl,
            emb_bytes,
        )
        await redis.setex(
            f"{CACHE_KEY_PREFIX}{q_hash}",
            ttl,
            json.dumps(response, default=str),
        )
        logger.info("cache_stored", query_hash=q_hash[:8])
    except Exception as e:
        logger.warning("cache_store_failed", error=str(e))


async def flush_semantic_cache() -> int:
    try:
        redis = get_redis()
        keys = await redis.keys(
            f"{CACHE_KEY_PREFIX}*"
        ) + await redis.keys(f"{EMBEDDING_KEY_PREFIX}*")
        if keys:
            await redis.delete(*keys)
        return len(keys)
    except Exception as e:
        logger.warning("cache_flush_failed", error=str(e))
        return 0


async def get_session(session_id: str) -> List[Dict[str, str]]:
    try:
        redis = get_redis()
        key = f"session:{session_id}"
        raw = await redis.get(key)
        if raw:
            await redis.expire(key, 1800)
            return json.loads(raw)
    except Exception as e:
        logger.warning("session_get_failed", error=str(e))
    return []


async def store_session(
    session_id: str,
    messages: List[Dict[str, str]],
    max_turns: int = 5,
) -> None:
    try:
        redis = get_redis()
        key = f"session:{session_id}"
        trimmed = messages[-(max_turns * 2):]
        await redis.setex(
            key,
            1800,
            json.dumps(trimmed),
        )
    except Exception as e:
        logger.warning("session_store_failed", error=str(e))


async def delete_session(session_id: str) -> None:
    try:
        redis = get_redis()
        await redis.delete(f"session:{session_id}")
    except Exception as e:
        logger.warning("session_delete_failed", error=str(e))


async def check_redis() -> bool:
    try:
        redis = get_redis()
        await redis.ping()
        return True
    except Exception:
        return False


async def increment_session_query_count(
    session_id: str,
) -> int:
    try:
        redis = get_redis()
        key = f"session_qcount:{session_id}"
        count = await redis.incr(key)
        await redis.expire(key, 86400)
        return count
    except Exception:
        return 0
