"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2026-03-10
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "providers",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "npi",
            sa.String(10),
            unique=True,
            nullable=False,
        ),
        sa.Column(
            "name", sa.String(120), nullable=False
        ),
        sa.Column(
            "gender", sa.String(1), nullable=False
        ),
        sa.Column(
            "specialties",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "state", sa.String(2), nullable=False
        ),
        sa.Column(
            "city", sa.String(80), nullable=False
        ),
        sa.Column(
            "address", sa.String(200), nullable=False
        ),
        sa.Column(
            "lat",
            sa.Numeric(9, 6),
            nullable=False,
        ),
        sa.Column(
            "long",
            sa.Numeric(9, 6),
            nullable=False,
        ),
        sa.Column(
            "insurances",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "rating",
            sa.Numeric(3, 2),
            nullable=True,
        ),
        sa.Column(
            "accepting_new_patients",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column("bio", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )

    op.execute("""
        ALTER TABLE providers
        ADD COLUMN bio_tsv TSVECTOR
        GENERATED ALWAYS AS (to_tsvector('english', bio)) STORED
    """)

    op.create_index(
        "idx_providers_state",
        "providers",
        ["state"],
    )
    op.create_index(
        "idx_providers_specialties",
        "providers",
        ["specialties"],
        postgresql_using="gin",
    )
    op.create_index(
        "idx_providers_insurances",
        "providers",
        ["insurances"],
        postgresql_using="gin",
    )
    op.execute(
        "CREATE INDEX idx_providers_bio_tsv "
        "ON providers USING gin (bio_tsv)"
    )

    op.create_table(
        "embeddings",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "provider_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "providers.id", ondelete="CASCADE"
            ),
            nullable=False,
        ),
        sa.Column(
            "chunk_index",
            sa.SmallInteger(),
            nullable=False,
        ),
        sa.Column(
            "content", sa.Text(), nullable=False
        ),
        sa.Column(
            "token_count",
            sa.SmallInteger(),
            nullable=False,
        ),
        sa.Column(
            "model",
            sa.String(60),
            nullable=False,
            server_default="text-embedding-3-small",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "provider_id",
            "chunk_index",
            name="uq_embeddings_provider_chunk",
        ),
    )

    op.execute(
        "ALTER TABLE embeddings "
        "ADD COLUMN embedding vector(1536) NOT NULL "
        "DEFAULT array_fill(0, ARRAY[1536])::vector"
    )
    op.execute(
        "ALTER TABLE embeddings "
        "ALTER COLUMN embedding DROP DEFAULT"
    )
    op.execute(
        "CREATE INDEX idx_embeddings_hnsw "
        "ON embeddings "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )

    op.create_table(
        "audit_logs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column(
            "query_hash",
            sa.String(64),
            nullable=False,
        ),
        sa.Column(
            "latency_ms", sa.Integer(), nullable=True
        ),
        sa.Column(
            "token_in", sa.Integer(), nullable=True
        ),
        sa.Column(
            "token_out", sa.Integer(), nullable=True
        ),
        sa.Column(
            "cache_hit",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_audit_created",
        "audit_logs",
        ["created_at"],
    )

    op.create_table(
        "ingest_jobs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "status",
            sa.String(20),
            server_default="pending",
            nullable=False,
        ),
        sa.Column(
            "total", sa.Integer(), nullable=True
        ),
        sa.Column(
            "processed",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "errors",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "finished_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "error_msg", sa.Text(), nullable=True
        ),
    )


def downgrade() -> None:
    op.drop_table("ingest_jobs")
    op.drop_table("audit_logs")
    op.drop_table("embeddings")
    op.drop_table("providers")
