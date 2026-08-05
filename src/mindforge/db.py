"""Database layer — PostgreSQL via SQLAlchemy."""

from __future__ import annotations

import os
import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Load the same project-root .env used by the application configuration.
try:
    from dotenv import load_dotenv as _load_dotenv
    from mindforge.config import get_project_root

    _env_override = os.getenv("MINDFORGE_ENV_FILE", "").strip()
    _env = (
        Path(_env_override).expanduser().resolve()
        if _env_override
        else get_project_root() / ".env"
    )
    if _env.is_file():
        _load_dotenv(str(_env), encoding="utf-8")
except Exception:
    pass

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    inspect,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from cryptography.fernet import Fernet, InvalidToken
from mindforge.config import require_environment_variable

# ---------------------------------------------------------------------------
# Engine — PostgreSQL only
# ---------------------------------------------------------------------------

_DB_URL = require_environment_variable("DATABASE_URL")

_engine = create_engine(
    _DB_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    connect_args={"connect_timeout": 5},
)

SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
# ---------------------------------------------------------------------------
# Encrypted API key helpers. New values use Fernet; legacy XOR values remain
# readable only for migration compatibility.
# ---------------------------------------------------------------------------

_SECRET_WARNED = False
_SECRET_CACHE: str | None = None


def _persist_generated_secret(secret: str) -> bool:
    """Persist a generated APP_SECRET to the project env file when possible."""
    env_override = os.getenv("MINDFORGE_ENV_FILE")
    env_path = (
        Path(env_override).expanduser() if env_override else get_project_root() / ".env"
    )
    try:
        from dotenv import set_key

        env_path.parent.mkdir(parents=True, exist_ok=True)
        if not env_path.exists():
            env_path.touch(mode=0o600)
        set_key(str(env_path), "APP_SECRET", secret, quote_mode="never")
        return True
    except Exception:
        return False


def _get_secret() -> bytes:
    global _SECRET_CACHE, _SECRET_WARNED
    secret = os.getenv("APP_SECRET", "") or _SECRET_CACHE or ""
    if not secret:
        secret = secrets.token_hex(32)
        _SECRET_CACHE = secret
        os.environ["APP_SECRET"] = secret
        persisted = _persist_generated_secret(secret)
        if not _SECRET_WARNED:
            import logging

            logging.getLogger(__name__).warning(
                "APP_SECRET was not set; generated %s secret.",
                "and persisted a" if persisted else "an in-process",
            )
            _SECRET_WARNED = True
    else:
        _SECRET_CACHE = secret
    return hashlib.sha256(secret.encode()).digest()


def encrypt_api_key(plain: str) -> str:
    """Encrypt an API key with authenticated Fernet encryption."""
    if not plain:
        return ""
    key = base64.urlsafe_b64encode(_get_secret())
    token = Fernet(key).encrypt(plain.encode("utf-8")).decode("ascii")
    return f"fernet:{token}"


def decrypt_api_key(encrypted: str) -> str:
    """Decrypt current Fernet values and legacy XOR-hex values."""
    if not encrypted:
        return ""
    secret = _get_secret()
    try:
        if encrypted.startswith("fernet:"):
            key = base64.urlsafe_b64encode(secret)
            token = encrypted.removeprefix("fernet:")
            return Fernet(key).decrypt(token.encode("ascii")).decode("utf-8")

        # Backward compatibility for values created by older MindForge builds.
        raw = bytes.fromhex(encrypted)
        decrypted = bytes(b ^ secret[i % len(secret)] for i, b in enumerate(raw))
        return decrypted.decode("utf-8")
    except (InvalidToken, ValueError, UnicodeDecodeError):
        return ""


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    @staticmethod
    def hash_password(password: str) -> str:
        salt = secrets.token_hex(16)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
        return salt + ":" + dk.hex()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        try:
            salt, h = hashed.split(":", 1)
            candidate = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), salt.encode(), 100000
            ).hex()
            return hmac.compare_digest(h, candidate)
        except Exception:
            return False


class ApiKey(Base):
    __tablename__ = "api_keys"
    __table_args__ = (
        UniqueConstraint("user_id", "provider", name="uq_api_keys_user_provider"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    provider: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # Stable LLM provider identifier.
    key_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    base_url: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class ResearchHistory(Base):
    __tablename__ = "research_history"
    __table_args__ = (
        Index(
            "ix_research_history_user_created_id",
            "user_id",
            "created_at",
            "id",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    task: Mapped[str] = mapped_column(Text, nullable=False)
    report: Mapped[str] = mapped_column(Text, nullable=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_used: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    token_usage: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    sources: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    trace_id: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
        index=True,
    )
    conversation_id: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
        index=True,
    )
    run_id: Mapped[Optional[str]] = mapped_column(
        String(80),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class DocumentCatalog(Base):
    """Persistent document metadata independent of Qdrant chunk payloads."""

    __tablename__ = "document_catalog"

    doc_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="indexing",
        index=True,
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )
    index_signature: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )
    index_strategy: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="auto",
    )
    use_raptor: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    use_graphrag: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parser_metadata: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class IndexJob(Base):
    """Persistent state for asynchronous document indexing."""

    __tablename__ = "index_jobs"
    __table_args__ = (Index("ix_index_jobs_status_created", "status", "created_at"),)

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="queued",
    )
    stage: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="queued",
    )
    progress: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    timings: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    metrics: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    strategy: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="auto",
    )
    use_raptor: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    use_graphrag: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class DocumentAsset(Base):
    """Persistent visual and structural assets emitted during document parsing."""

    __tablename__ = "document_assets"
    __table_args__ = (
        Index("ix_document_assets_doc_page", "doc_id", "page"),
        Index("ix_document_assets_doc_element", "doc_id", "element_index"),
    )

    asset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        ForeignKey("document_catalog.doc_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    element_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    relative_path: Mapped[Optional[str]] = mapped_column(
        String(1024),
        nullable=True,
    )
    content_type: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
    )
    sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class Conversation(Base):
    """User-visible research conversation."""

    __tablename__ = "conversations"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'archived', 'deleted')",
            name="ck_conversations_status",
        ),
        CheckConstraint(
            "context_mode IN ('auto', 'manual', 'disabled')",
            name="ck_conversations_context_mode",
        ),
        CheckConstraint("next_sequence >= 1", name="ck_conversations_sequence"),
        CheckConstraint("version >= 1", name="ck_conversations_version"),
        Index(
            "ix_conversations_user_status_updated",
            "user_id",
            "status",
            "updated_at",
        ),
    )

    conversation_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="active",
    )
    context_mode: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="auto",
    )
    next_sequence: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=1,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class ConversationMessage(Base):
    """One visible message in a research conversation."""

    __tablename__ = "conversation_messages"
    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant', 'system_notice')",
            name="ck_conversation_messages_role",
        ),
        CheckConstraint(
            "context_scope IN ('turn', 'conversation', 'user')",
            name="ck_conversation_messages_scope",
        ),
        CheckConstraint("sequence >= 1", name="ck_conversation_messages_sequence"),
        UniqueConstraint(
            "conversation_id",
            "sequence",
            name="uq_conversation_messages_sequence",
        ),
        Index(
            "ix_conversation_messages_context",
            "conversation_id",
            "include_in_context",
            "deleted_at",
            "sequence",
        ),
        Index("ix_conversation_messages_run", "run_id"),
    )

    message_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=False,
    )
    run_id: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    include_in_context: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )
    pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    context_scope: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="conversation",
    )
    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    edited_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class ConversationSummary(Base):
    """Structured compression of an earlier conversation range."""

    __tablename__ = "conversation_summaries"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'invalidated', 'superseded')",
            name="ck_conversation_summaries_status",
        ),
        CheckConstraint(
            "from_sequence >= 1 AND to_sequence >= from_sequence",
            name="ck_conversation_summaries_range",
        ),
        Index(
            "ix_conversation_summaries_active",
            "conversation_id",
            "status",
            "to_sequence",
        ),
    )

    summary_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=False,
    )
    from_sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    to_sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    summary: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    source_message_ids: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="active",
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class ResearchArtifact(Base):
    """Fine-grained, reusable output from a completed research run."""

    __tablename__ = "research_artifacts"
    __table_args__ = (
        CheckConstraint(
            "artifact_type IN ('subtask_finding', 'report_section', 'claim', "
            "'evidence', 'decision', 'limitation', 'citation_source')",
            name="ck_research_artifacts_type",
        ),
        CheckConstraint(
            "grounding_status IN ('grounded', 'model_only', 'not_required')",
            name="ck_research_artifacts_grounding",
        ),
        CheckConstraint(
            "freshness_class IN ('stable', 'time_sensitive', 'volatile')",
            name="ck_research_artifacts_freshness",
        ),
        Index(
            "ix_research_artifacts_recall",
            "conversation_id",
            "enabled",
            "deleted_at",
            "created_at",
        ),
        Index("ix_research_artifacts_run", "run_id"),
    )

    artifact_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    run_id: Mapped[str] = mapped_column(String(80), nullable=False)
    conversation_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=True,
    )
    artifact_type: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    grounding_status: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        default="grounded",
    )
    freshness_class: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        default="stable",
    )
    valid_from: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    embedding_version: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default="lexical-v1",
    )
    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class ContextSnapshot(Base):
    """Immutable record of context selected for one research run."""

    __tablename__ = "context_snapshots"
    __table_args__ = (
        CheckConstraint(
            "budget_tokens >= 0 AND used_tokens >= 0 "
            "AND used_tokens <= budget_tokens",
            name="ck_context_snapshots_budget",
        ),
        UniqueConstraint("run_id", name="uq_context_snapshots_run"),
        Index("ix_context_snapshots_conversation", "conversation_id", "created_at"),
        Index("ix_context_snapshots_created_at", "created_at"),
    )

    snapshot_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(80), nullable=False)
    conversation_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("conversations.conversation_id", ondelete="SET NULL"),
        nullable=True,
    )
    query_message_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("conversation_messages.message_id", ondelete="SET NULL"),
        nullable=True,
    )
    standalone_query: Mapped[str] = mapped_column(Text, nullable=False)
    context_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    budget_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    used_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    policy_version: Mapped[str] = mapped_column(String(40), nullable=False)
    embedding_version: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class ContextSnapshotItem(Base):
    """One immutable item included in a context snapshot."""

    __tablename__ = "context_snapshot_items"
    __table_args__ = (
        CheckConstraint("rank >= 1", name="ck_context_snapshot_items_rank"),
        CheckConstraint(
            "token_count >= 0",
            name="ck_context_snapshot_items_tokens",
        ),
        CheckConstraint(
            "freshness_status IN ('current', 'stale', 'expired')",
            name="ck_context_snapshot_items_freshness",
        ),
        UniqueConstraint(
            "snapshot_id",
            "rank",
            name="uq_context_snapshot_items_rank",
        ),
        Index(
            "ix_context_snapshot_items_source",
            "source_type",
            "source_id",
        ),
    )

    snapshot_item_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("context_snapshots.snapshot_id", ondelete="CASCADE"),
        nullable=False,
    )
    source_type: Mapped[str] = mapped_column(String(24), nullable=False)
    source_id: Mapped[str] = mapped_column(String(64), nullable=False)
    content_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    selection_reason: Mapped[str] = mapped_column(String(200), nullable=False)
    pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    freshness_status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="current",
    )
    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
    )


class UserMemory(Base):
    """User-managed cross-conversation memory."""

    __tablename__ = "user_memories"
    __table_args__ = (
        CheckConstraint(
            "category IN ('preference', 'profile', 'stable_fact', "
            "'project_context', 'decision')",
            name="ck_user_memories_category",
        ),
        CheckConstraint(
            "status IN ('active', 'superseded', 'forgotten')",
            name="ck_user_memories_status",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_user_memories_confidence",
        ),
        Index(
            "ix_user_memories_active",
            "user_id",
            "status",
            "deleted_at",
            "updated_at",
        ),
    )

    memory_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    category: Mapped[str] = mapped_column(String(24), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_message_ids: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="active",
    )
    user_confirmed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    valid_from: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class ContextLineage(Base):
    """Dependency edge used for deletion and invalidation propagation."""

    __tablename__ = "context_lineage"
    __table_args__ = (
        CheckConstraint(
            "source_type IN ('message', 'artifact', 'memory')",
            name="ck_context_lineage_source_type",
        ),
        CheckConstraint(
            "derived_type IN ('summary', 'artifact', 'memory', 'cache')",
            name="ck_context_lineage_derived_type",
        ),
        UniqueConstraint(
            "source_type",
            "source_id",
            "derived_type",
            "derived_id",
            "relation",
            name="uq_context_lineage_edge",
        ),
        Index("ix_context_lineage_source", "source_type", "source_id"),
        Index("ix_context_lineage_derived", "derived_type", "derived_id"),
    )

    lineage_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    source_type: Mapped[str] = mapped_column(String(24), nullable=False)
    source_id: Mapped[str] = mapped_column(String(64), nullable=False)
    derived_type: Mapped[str] = mapped_column(String(24), nullable=False)
    derived_id: Mapped[str] = mapped_column(String(64), nullable=False)
    relation: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class DeletionJob(Base):
    """Persistent status for context deletion propagation."""

    __tablename__ = "deletion_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_deletion_jobs_status",
        ),
        Index("ix_deletion_jobs_user_created", "user_id", "created_at"),
    )

    deletion_job_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_type: Mapped[str] = mapped_column(String(24), nullable=False)
    target_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


def _alembic_config():
    from alembic.config import Config
    from mindforge.config import get_project_root

    config = Config()
    config.set_main_option(
        "script_location",
        str(get_project_root() / "migrations"),
    )
    config.set_main_option("sqlalchemy.url", _DB_URL.replace("%", "%%"))
    return config


def run_migrations() -> None:
    """Upgrade the database, safely adopting pre-Alembic installations."""
    from alembic import command

    config = _alembic_config()
    with _engine.begin() as connection:
        config.attributes["connection"] = connection
        tables = set(inspect(connection).get_table_names())
        legacy_core = {"users", "api_keys", "research_history"}
        if "alembic_version" not in tables and legacy_core.issubset(tables):
            command.stamp(config, "0001_baseline")
        command.upgrade(config, "head")


def init_db() -> None:
    """Apply schema migrations and ensure the single-user account exists."""
    run_migrations()

    # Ensure default user exists
    with SessionLocal() as db:
        user = db.query(User).filter(User.username == "default").first()
        if user is None:
            user = User(
                username="default",
                password_hash=User.hash_password("mindforge"),
            )
            db.add(user)
            db.commit()


def get_default_user_id(db: Session) -> int:
    """Return the actual id of the single-user account, creating it if needed."""
    user = db.query(User).filter(User.username == "default").first()
    if user is None:
        user = User(
            username="default",
            password_hash=User.hash_password("mindforge"),
        )
        db.add(user)
        db.flush()
    return user.id
