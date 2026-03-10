import hashlib
import html
import re
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

INJECTION_PATTERNS = re.compile(
    r"(</?\s*system\s*>|<\|im_start\|>|<\|im_end\|>|"
    r"ignore\s+previous\s+instructions|"
    r"\bDAN\b|jailbreak|"
    r"pretend\s+you\s+are|"
    r"you\s+are\s+now|"
    r"disregard\s+(all\s+)?(previous|prior|above))",
    re.IGNORECASE,
)

STOPWORDS_ONLY_PATTERN = re.compile(
    r"^\s*(the|a|an|is|are|was|were|be|been|"
    r"being|have|has|had|do|does|did|will|"
    r"would|could|should|may|might|shall|"
    r"and|or|but|if|in|on|at|to|for|of|"
    r"with|by|from|up|about|into|through"
    r"[\s,]+)*\s*$",
    re.IGNORECASE,
)

MAX_QUERY_TOKENS = 500


def sanitise_query(raw: str) -> str:
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def check_injection(text: str) -> bool:
    return bool(INJECTION_PATTERNS.search(text))


def is_stopwords_only(text: str) -> bool:
    return bool(STOPWORDS_ONLY_PATTERN.match(text.strip()))


def scrub_pii(text: str) -> str:
    try:
        from presidio_analyzer import AnalyzerEngine

        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text=text, language="en")
        if results:
            logger.warning(
                "pii_detected",
                entities=[r.entity_type for r in results],
            )
            for result in sorted(
                results, key=lambda x: x.start, reverse=True
            ):
                text = (
                    text[: result.start]
                    + f"[{result.entity_type}]"
                    + text[result.end :]
                )
    except Exception:
        pass
    return text


def detect_language(text: str) -> Optional[str]:
    try:
        from langdetect import detect

        return detect(text)
    except Exception:
        return None


def hash_query(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_admin_key(provided: str, expected: str) -> bool:
    return hashlib.compare_digest(
        provided.encode(), expected.encode()
    )


def escape_output(text: str) -> str:
    return html.escape(text)
