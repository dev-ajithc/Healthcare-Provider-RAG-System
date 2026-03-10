import uuid
from datetime import datetime
from typing import List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Provider(Base):
    __tablename__ = "providers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    npi: Mapped[str] = mapped_column(
        String(10), unique=True, nullable=False
    )
    name: Mapped[str] = mapped_column(
        String(120), nullable=False
    )
    gender: Mapped[str] = mapped_column(
        String(1), nullable=False
    )
    specialties: Mapped[List[str]] = mapped_column(
        ARRAY(Text), nullable=False
    )
    state: Mapped[str] = mapped_column(
        String(2), nullable=False, index=True
    )
    city: Mapped[str] = mapped_column(String(80), nullable=False)
    address: Mapped[str] = mapped_column(
        String(200), nullable=False
    )
    lat: Mapped[float] = mapped_column(
        Numeric(9, 6), nullable=False
    )
    long: Mapped[float] = mapped_column(
        Numeric(9, 6), nullable=False
    )
    insurances: Mapped[List[str]] = mapped_column(
        ARRAY(Text), nullable=False
    )
    rating: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2), nullable=True
    )
    accepting_new_patients: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )
    bio: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    provider_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("providers.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(
        SmallInteger, nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[List[float]] = mapped_column(
        Vector(1536), nullable=False
    )
    token_count: Mapped[int] = mapped_column(
        SmallInteger, nullable=False
    )
    model: Mapped[str] = mapped_column(
        String(60),
        nullable=False,
        default="text-embedding-3-small",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    query_hash: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    latency_ms: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    token_in: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    token_out: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    cache_hit: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class IngestJob(Base):
    __tablename__ = "ingest_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    status: Mapped[str] = mapped_column(
        String(20), default="pending", nullable=False
    )
    total: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    processed: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    errors: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_msg: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
