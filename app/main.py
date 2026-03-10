import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.api import health, ingest, query, session
from app.core.config import settings
from app.core.middleware import (
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    TimingMiddleware,
)
from app.core.telemetry import setup_telemetry

logger = structlog.get_logger(__name__)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[
        f"{settings.rate_limit_per_minute}/minute"
    ],
)


def create_app() -> FastAPI:
    setup_telemetry()

    app = FastAPI(
        title="Healthcare Provider RAG System",
        description=(
            "AI-powered natural language search over synthetic "
            "healthcare provider data."
        ),
        version=settings.app_version,
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url="/redoc" if settings.app_env != "production" else None,
    )

    app.state.limiter = limiter
    app.add_exception_handler(
        RateLimitExceeded, _rate_limit_exceeded_handler
    )

    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    if settings.app_env == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*.onrender.com", "localhost"],
        )

    app.include_router(query.router, tags=["Query"])
    app.include_router(health.router, tags=["Health"])
    app.include_router(session.router, tags=["Session"])
    app.include_router(ingest.router, tags=["Ingest"])

    @app.on_event("startup")
    async def startup() -> None:
        logger.info(
            "app_startup",
            version=settings.app_version,
            env=settings.app_env,
        )

    @app.on_event("shutdown")
    async def shutdown() -> None:
        logger.info("app_shutdown")

    return app


app = create_app()
