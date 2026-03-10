import time
import uuid
from typing import Any, Dict, List, Optional

import structlog

from app.core.config import settings
from app.core.security import (
    check_injection,
    detect_language,
    hash_query,
    is_stopwords_only,
    sanitise_query,
    scrub_pii,
)
from app.core.telemetry import (
    CACHE_HIT_RATIO,
    HALLUCINATION_TOTAL,
    LLM_TOKENS,
    QUERY_LATENCY,
    QUERY_TOTAL,
    RETRIEVAL_HITS,
    get_tracer,
)
from app.db.queries import (
    hybrid_retrieve,
    insert_audit_log,
)
from app.db.session import AsyncSessionLocal
from app.rag.cache import (
    delete_session,
    get_session,
    increment_session_query_count,
    semantic_cache_lookup,
    semantic_cache_store,
    store_session,
)
from app.rag.embeddings import embed_text
from app.rag.llm import generate
from app.rag.retrieval import (
    build_context,
    extract_intent_filters,
    rerank,
)
from app.schemas import (
    ProviderMapPoint,
    QueryRequest,
    QueryResponse,
    SourceItem,
)

logger = structlog.get_logger(__name__)
_cache_hits = 0
_cache_total = 0


async def process_query(
    request: QueryRequest,
    request_id: Optional[str] = None,
) -> QueryResponse:
    global _cache_hits, _cache_total
    start_time = time.perf_counter()
    tracer = get_tracer()

    if settings.query_killswitch:
        raise RuntimeError("QUERY_KILLSWITCH is active")

    session_id = request.session_id or uuid.uuid4()

    with tracer.start_as_current_span("process_query") as span:
        span.set_attribute("session_id", str(session_id))
        span.set_attribute(
            "request_id", request_id or ""
        )

        with tracer.start_as_current_span("sanitise_query"):
            clean = sanitise_query(request.query)

            if is_stopwords_only(clean):
                return _empty_response(
                    session_id,
                    "Please provide more specific search terms.",
                    [
                        "Search by specialty (e.g., cardiologist)",
                        "Search by location (e.g., California)",
                        "Search by insurance (e.g., Medicare)",
                    ],
                )

            lang = detect_language(clean)
            if lang and lang != "en":
                return _empty_response(
                    session_id,
                    "Please query in English.",
                    [],
                )

            if check_injection(clean):
                logger.warning(
                    "injection_detected",
                    request_id=request_id,
                    session_id=str(session_id),
                )
                QUERY_TOTAL.labels(status="injection").inc()
                return _empty_response(
                    session_id,
                    "Your query could not be processed.",
                    [],
                )

            clean = scrub_pii(clean)

        with tracer.start_as_current_span("embed_query"):
            query_embedding = await embed_text(clean)

        with tracer.start_as_current_span("cache_lookup"):
            cached_resp, cache_hit = (
                await semantic_cache_lookup(query_embedding)
            )
            _cache_total += 1
            if cache_hit:
                _cache_hits += 1

        CACHE_HIT_RATIO.set(
            _cache_hits / _cache_total if _cache_total > 0 else 0
        )

        if cache_hit and cached_resp:
            latency_ms = int(
                (time.perf_counter() - start_time) * 1000
            )
            QUERY_LATENCY.labels(
                cache_hit="true", model="cached"
            ).observe(latency_ms / 1000)
            QUERY_TOTAL.labels(status="cached").inc()

            return _build_response(
                cached_resp,
                session_id,
                latency_ms,
                cache_hit=True,
                model_used="cached",
            )

        count = await increment_session_query_count(
            str(session_id)
        )
        if count > settings.max_session_queries:
            return _empty_response(
                session_id,
                "Session query limit reached. Please start a new session.",
                [],
            )

        session_history = await get_session(str(session_id))

        if request.hyde_enabled or settings.hyde_enabled:
            with tracer.start_as_current_span("hyde"):
                clean, query_embedding = await _apply_hyde(
                    clean, query_embedding
                )

        intent_filters = extract_intent_filters(clean)
        if request.filters:
            if intent_filters is None:
                intent_filters = {}
            intent_filters.update(request.filters)

        with tracer.start_as_current_span("hybrid_retrieve"):
            async with AsyncSessionLocal() as db:
                candidates = await hybrid_retrieve(
                    db=db,
                    query_embedding=query_embedding,
                    query_text=clean,
                    top_k=settings.retrieve_top_k,
                    similarity_cutoff=settings.similarity_cutoff,
                    filters=intent_filters,
                )
            span.set_attribute(
                "retrieval.candidates", len(candidates)
            )
            RETRIEVAL_HITS.observe(len(candidates))

        if not candidates:
            return _empty_response(
                session_id,
                "No matching providers found. "
                "Try adjusting your search criteria.",
                [
                    "Broaden your specialty search",
                    "Try a different state",
                    "Remove insurance filter",
                ],
            )

        with tracer.start_as_current_span("rerank"):
            top_chunks = rerank(
                clean,
                candidates,
                top_k=settings.rerank_top_k,
            )

        context = build_context(top_chunks)

        with tracer.start_as_current_span("llm_call"):
            llm_data, model_used, tok_in, tok_out = (
                await generate(
                    clean,
                    context,
                    history=session_history,
                )
            )
            LLM_TOKENS.labels(direction="in").inc(tok_in)
            LLM_TOKENS.labels(direction="out").inc(tok_out)

        with tracer.start_as_current_span("validate_output"):
            hallucinated = _check_hallucination(
                llm_data, top_chunks
            )
            if hallucinated:
                HALLUCINATION_TOTAL.inc()
                logger.error(
                    "hallucination_detected",
                    session_id=str(session_id),
                )
                llm_data["_hallucination_warning"] = True

        await semantic_cache_store(
            clean, query_embedding, llm_data
        )

        new_history = session_history + [
            {"role": "user", "content": clean},
            {
                "role": "assistant",
                "content": llm_data.get("answer", ""),
            },
        ]
        await store_session(str(session_id), new_history)

        latency_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        async with AsyncSessionLocal() as db:
            await insert_audit_log(
                db=db,
                session_id=session_id,
                query_hash=hash_query(clean),
                latency_ms=latency_ms,
                token_in=tok_in,
                token_out=tok_out,
                cache_hit=False,
            )

        QUERY_LATENCY.labels(
            cache_hit="false", model=model_used
        ).observe(latency_ms / 1000)
        QUERY_TOTAL.labels(status="ok").inc()

        logger.info(
            "query_completed",
            latency_ms=latency_ms,
            cache_hit=False,
            tokens_in=tok_in,
            tokens_out=tok_out,
            model=model_used,
            retrieval_hits=len(top_chunks),
        )

        return _build_response(
            llm_data,
            session_id,
            latency_ms,
            cache_hit=False,
            model_used=model_used,
        )


async def _apply_hyde(
    query: str,
    original_embedding: List[float],
) -> "tuple[str, List[float]]":
    try:
        from app.rag.llm import get_anthropic_client

        client = get_anthropic_client()
        response = await client.messages.create(
            model=settings.llm_primary,
            max_tokens=150,
            system=(
                "Generate a one-sentence description of an ideal "
                "healthcare provider that would answer this query. "
                "Output only the description, no preamble."
            ),
            messages=[{"role": "user", "content": query}],
        )
        hypothetical = response.content[0].text.strip()
        hypo_embedding = await embed_text(hypothetical)
        return hypothetical, hypo_embedding
    except Exception as e:
        logger.warning("hyde_failed", error=str(e))
        return query, original_embedding


def _check_hallucination(
    llm_data: Dict[str, Any],
    top_chunks: List[Dict[str, Any]],
) -> bool:
    chunk_npis = {
        str(c.get("npi", "")) for c in top_chunks
    }
    answer = llm_data.get("answer", "")
    for source in llm_data.get("sources", []):
        npi = str(source.get("npi", ""))
        if npi and npi not in chunk_npis:
            return True
        snippet = source.get("snippet", "")
        if snippet:
            found = any(
                snippet[:30] in c.get("content", "")
                for c in top_chunks
            )
            if not found and snippet[:30]:
                return True
    import re

    npi_pattern = re.compile(r"\b\d{10}\b")
    answer_npis = set(npi_pattern.findall(answer))
    for npi in answer_npis:
        if npi not in chunk_npis:
            return True
    return False


def _build_response(
    llm_data: Dict[str, Any],
    session_id: uuid.UUID,
    latency_ms: int,
    cache_hit: bool,
    model_used: str,
) -> QueryResponse:
    sources = []
    for item in llm_data.get("sources", [])[:10]:
        try:
            sources.append(
                SourceItem(
                    id=int(item.get("id", 0)),
                    npi=str(item.get("npi", "")),
                    provider_name=str(
                        item.get("provider_name", "")
                    ),
                    snippet=str(item.get("snippet", "")),
                    relevance_score=float(
                        item.get("relevance_score", 0.0)
                    ),
                )
            )
        except Exception:
            continue

    map_points = []
    for pt in llm_data.get("providers_map", [])[:20]:
        try:
            map_points.append(
                ProviderMapPoint(
                    npi=str(pt.get("npi", "")),
                    lat=float(pt.get("lat", 0.0)),
                    long=float(pt.get("long", 0.0)),
                    name=str(pt.get("name", "")),
                    specialty=str(pt.get("specialty", "")),
                )
            )
        except Exception:
            continue

    answer = llm_data.get("answer", "")
    if llm_data.get("_hallucination_warning"):
        answer = (
            "> ⚠️ **Warning:** Some citations may not be "
            "fully verifiable. Please review sources carefully.\n\n"
            + answer
        )

    return QueryResponse(
        answer=answer,
        sources=sources,
        suggestions=llm_data.get("suggestions", [])[:5],
        providers_map=map_points,
        session_id=session_id,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        model_used=model_used,
    )


def _empty_response(
    session_id: uuid.UUID,
    message: str,
    suggestions: List[str],
) -> QueryResponse:
    QUERY_TOTAL.labels(status="no_results").inc()
    return QueryResponse(
        answer=message,
        sources=[],
        suggestions=suggestions,
        providers_map=[],
        session_id=session_id,
        latency_ms=0,
        cache_hit=False,
        model_used="none",
    )


async def get_session_history(
    session_id: str,
) -> List[Dict[str, str]]:
    return await get_session(session_id)


async def clear_session(session_id: str) -> None:
    await delete_session(session_id)
