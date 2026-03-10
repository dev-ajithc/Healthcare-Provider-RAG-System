import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    session_id: Optional[uuid.UUID] = None
    filters: Optional[Dict[str, Any]] = None
    hyde_enabled: bool = False


class SourceItem(BaseModel):
    id: int
    npi: str
    provider_name: str
    snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class ProviderMapPoint(BaseModel):
    npi: str
    lat: float
    long: float
    name: str
    specialty: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    suggestions: List[str] = []
    providers_map: List[ProviderMapPoint] = []
    session_id: uuid.UUID
    latency_ms: int
    cache_hit: bool
    model_used: str


class HealthResponse(BaseModel):
    status: str
    db: str
    redis: str
    embed_api: str
    version: str
    uptime_seconds: float


class IngestStatusResponse(BaseModel):
    job_id: Optional[str]
    status: str
    total: Optional[int]
    processed: int
    errors: int
    started_at: Optional[str]
    finished_at: Optional[str]
    error_msg: Optional[str]


class SessionMessage(BaseModel):
    role: str
    content: str


class SessionResponse(BaseModel):
    session_id: uuid.UUID
    messages: List[SessionMessage]


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: Optional[str] = None
    retry_after: Optional[int] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
