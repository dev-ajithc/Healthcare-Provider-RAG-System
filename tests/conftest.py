"""Pytest configuration and shared fixtures."""

import os

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test_key")
os.environ.setdefault("OPENAI_API_KEY", "test_key")
os.environ.setdefault("LANGCHAIN_API_KEY", "test_key")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://rag:rag@localhost:5432/ragdb",
)
os.environ.setdefault(
    "REDIS_URL", "redis://localhost:6379/0"
)
os.environ.setdefault("ADMIN_KEY", "test_admin_key")
os.environ.setdefault("APP_ENV", "testing")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("OTLP_ENDPOINT", "http://localhost:4317")


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"
