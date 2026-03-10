from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    anthropic_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")

    # LangSmith
    langchain_api_key: str = Field(default="")
    langchain_tracing_v2: bool = Field(default=False)
    langchain_project: str = Field(
        default="healthcare-provider-rag"
    )

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://rag:rag@localhost:5432/ragdb"
    )

    # Redis
    redis_url: str = Field(
        default="redis://:secret@localhost:6379/0"
    )

    # Admin
    admin_key: str = Field(default="change_me_in_production")

    # Application
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    app_version: str = Field(default="1.0.0")

    # Feature flags
    hyde_enabled: bool = Field(default=False)
    query_killswitch: bool = Field(default=False)

    # Limits
    max_daily_tokens: int = Field(default=5_000_000)
    cache_ttl_seconds: int = Field(default=86400)
    max_session_queries: int = Field(default=50)
    rate_limit_per_minute: int = Field(default=10)
    similarity_cutoff: float = Field(default=0.5)
    cache_similarity_threshold: float = Field(default=0.97)
    retrieve_top_k: int = Field(default=20)
    rerank_top_k: int = Field(default=5)
    max_query_tokens: int = Field(default=500)
    max_output_tokens: int = Field(default=1500)

    # CORS
    allowed_origins: List[str] = Field(
        default=["http://localhost:8501"]
    )

    # OTLP
    otlp_endpoint: str = Field(
        default="http://localhost:4317"
    )

    # LLM models (pinned)
    embedding_model: str = Field(
        default="text-embedding-3-small"
    )
    embedding_dimensions: int = Field(default=1536)
    llm_primary: str = Field(
        default="claude-sonnet-4-5"
    )
    llm_fallback: str = Field(default="gpt-4o-mini")

    # Re-ranking
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


settings = Settings()
