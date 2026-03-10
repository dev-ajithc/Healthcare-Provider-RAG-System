from typing import Any, Dict, List, Optional

import structlog
from sentence_transformers import CrossEncoder

from app.core.config import settings

logger = structlog.get_logger(__name__)

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(
            "loading_reranker", model=settings.reranker_model
        )
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = None,
) -> List[Dict[str, Any]]:
    if top_k is None:
        top_k = settings.rerank_top_k

    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(query, c["content"]) for c in candidates]

    try:
        scores = reranker.predict(pairs)
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])
        ranked = sorted(
            candidates,
            key=lambda x: x.get("rerank_score", 0.0),
            reverse=True,
        )
        return ranked[:top_k]
    except Exception as e:
        logger.warning("reranking_failed", error=str(e))
        return candidates[:top_k]


def build_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        specialties = chunk.get("specialties", [])
        if isinstance(specialties, list):
            spec_str = ", ".join(specialties)
        else:
            spec_str = str(specialties)

        insurances = chunk.get("insurances", [])
        if isinstance(insurances, list):
            ins_str = ", ".join(insurances)
        else:
            ins_str = str(insurances)

        rating = chunk.get("rating")
        rating_str = f"{rating}" if rating else "N/A"

        accepting = chunk.get("accepting_new_patients", False)
        accepting_str = "Yes" if accepting else "No"

        part = (
            f"[Source {i}]\n"
            f"Provider: {chunk.get('name', 'Unknown')} "
            f"(NPI: {chunk.get('npi', 'N/A')})\n"
            f"Specialty: {spec_str} | "
            f"State: {chunk.get('state', 'N/A')} | "
            f"Insurance: {ins_str}\n"
            f"Rating: {rating_str} | "
            f"Accepting New Patients: {accepting_str}\n"
            f"---\n"
            f"{chunk.get('content', '')}"
        )
        parts.append(part)
    return "\n\n".join(parts)


def extract_intent_filters(
    query: str,
) -> Optional[Dict[str, Any]]:
    import re

    filters: Dict[str, Any] = {}

    us_states = {
        "alabama": "AL", "alaska": "AK", "arizona": "AZ",
        "arkansas": "AR", "california": "CA", "colorado": "CO",
        "connecticut": "CT", "delaware": "DE", "florida": "FL",
        "georgia": "GA", "hawaii": "HI", "idaho": "ID",
        "illinois": "IL", "indiana": "IN", "iowa": "IA",
        "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
        "maine": "ME", "maryland": "MD", "massachusetts": "MA",
        "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
        "missouri": "MO", "montana": "MT", "nebraska": "NE",
        "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
        "new mexico": "NM", "new york": "NY",
        "north carolina": "NC", "north dakota": "ND",
        "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
        "pennsylvania": "PA", "rhode island": "RI",
        "south carolina": "SC", "south dakota": "SD",
        "tennessee": "TN", "texas": "TX", "utah": "UT",
        "vermont": "VT", "virginia": "VA", "washington": "WA",
        "west virginia": "WV", "wisconsin": "WI",
        "wyoming": "WY",
    }

    state_abbrevs = {v: v for v in us_states.values()}

    q_lower = query.lower()
    for name, abbrev in us_states.items():
        if name in q_lower:
            filters["state"] = abbrev
            break

    abbrev_match = re.search(
        r"\b([A-Z]{2})\b", query
    )
    if abbrev_match and not filters.get("state"):
        code = abbrev_match.group(1)
        if code in state_abbrevs:
            filters["state"] = code

    insurance_keywords = {
        "medicare": "Medicare",
        "medicaid": "Medicaid",
        "blue cross": "Blue Cross",
        "bcbs": "BCBS",
        "aetna": "Aetna",
        "cigna": "Cigna",
        "humana": "Humana",
        "united health": "UnitedHealth",
        "tricare": "Tricare",
    }
    for keyword, insurance in insurance_keywords.items():
        if keyword in q_lower:
            filters["insurance"] = insurance
            break

    accepting_patterns = [
        r"accepting new patients",
        r"taking new patients",
        r"available",
    ]
    for pattern in accepting_patterns:
        if re.search(pattern, q_lower):
            filters["accepting_new_patients"] = True
            break

    return filters if filters else None
