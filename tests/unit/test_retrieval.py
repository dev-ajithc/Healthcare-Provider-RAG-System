"""Unit tests for app.rag.retrieval module."""

from typing import Any, Dict, List

import pytest

from app.db.queries import _rrf_fusion
from app.rag.retrieval import (
    build_context,
    extract_intent_filters,
)


def _make_chunk(
    provider_id: str = "p1",
    chunk_index: int = 0,
    npi: str = "1234567890",
    name: str = "Dr. Test Provider",
    state: str = "CA",
    specialties: List[str] = None,
    insurances: List[str] = None,
    rating: float = 4.5,
    accepting: bool = True,
    content: str = "Test bio content for provider.",
    distance: float = 0.2,
) -> Dict[str, Any]:
    return {
        "provider_id": provider_id,
        "chunk_index": chunk_index,
        "npi": npi,
        "name": name,
        "state": state,
        "specialties": specialties or ["Cardiology"],
        "insurances": insurances or ["Medicare"],
        "rating": rating,
        "accepting_new_patients": accepting,
        "content": content,
        "distance": distance,
    }


class TestRRFFusion:
    def test_empty_both_inputs(self) -> None:
        result = _rrf_fusion([], [], top_k=5)
        assert result == []

    def test_dense_only(self) -> None:
        dense = [_make_chunk("p1", 0), _make_chunk("p2", 0)]
        result = _rrf_fusion(dense, [], top_k=5)
        assert len(result) == 2

    def test_bm25_only(self) -> None:
        bm25 = [_make_chunk("p1", 0), _make_chunk("p3", 0)]
        result = _rrf_fusion([], bm25, top_k=5)
        assert len(result) == 2

    def test_top_k_respected(self) -> None:
        dense = [_make_chunk(f"p{i}", 0) for i in range(10)]
        result = _rrf_fusion(dense, [], top_k=3)
        assert len(result) == 3

    def test_rrf_score_present(self) -> None:
        dense = [_make_chunk("p1", 0)]
        result = _rrf_fusion(dense, [], top_k=5)
        assert "rrf_score" in result[0]
        assert result[0]["rrf_score"] > 0


class TestBuildContext:
    def test_returns_string(self) -> None:
        chunks = [_make_chunk()]
        ctx = build_context(chunks)
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_source_numbering(self) -> None:
        chunks = [
            _make_chunk("p1", 0),
            _make_chunk("p2", 0),
        ]
        ctx = build_context(chunks)
        assert "[Source 1]" in ctx
        assert "[Source 2]" in ctx

    def test_npi_in_context(self) -> None:
        chunk = _make_chunk(npi="9876543210")
        ctx = build_context([chunk])
        assert "9876543210" in ctx

    def test_specialty_in_context(self) -> None:
        chunk = _make_chunk(specialties=["Pediatrics"])
        ctx = build_context([chunk])
        assert "Pediatrics" in ctx

    def test_insurance_in_context(self) -> None:
        chunk = _make_chunk(insurances=["Medicaid"])
        ctx = build_context([chunk])
        assert "Medicaid" in ctx

    def test_accepting_yes(self) -> None:
        chunk = _make_chunk(accepting=True)
        ctx = build_context([chunk])
        assert "Yes" in ctx

    def test_accepting_no(self) -> None:
        chunk = _make_chunk(accepting=False)
        ctx = build_context([chunk])
        assert "No" in ctx

    def test_empty_chunks_returns_empty(self) -> None:
        ctx = build_context([])
        assert ctx == ""


class TestExtractIntentFilters:
    def test_extracts_california(self) -> None:
        filters = extract_intent_filters(
            "cardiologists in california"
        )
        assert filters is not None
        assert filters.get("state") == "CA"

    def test_extracts_state_abbreviation(self) -> None:
        filters = extract_intent_filters(
            "doctors in TX accepting Medicare"
        )
        assert filters is not None
        assert filters.get("state") == "TX"

    def test_extracts_medicare(self) -> None:
        filters = extract_intent_filters(
            "providers accepting Medicare"
        )
        assert filters is not None
        assert filters.get("insurance") == "Medicare"

    def test_extracts_accepting_new_patients(self) -> None:
        filters = extract_intent_filters(
            "doctors accepting new patients in Florida"
        )
        assert filters is not None
        assert filters.get("accepting_new_patients") is True

    def test_no_filters_returns_none(self) -> None:
        filters = extract_intent_filters(
            "find me a good doctor"
        )
        assert filters is None

    def test_multiple_filters(self) -> None:
        filters = extract_intent_filters(
            "cardiologists in New York accepting Medicare"
        )
        assert filters is not None
        assert filters.get("state") == "NY"
        assert filters.get("insurance") == "Medicare"
