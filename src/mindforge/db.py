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
    Boolean,
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


def get_db() -> Session:
    """Return a new DB session. Caller must close it."""
    return SessionLocal()


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
        Path(env_override).expanduser()
        if env_override
        else get_project_root() / ".env"
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
    __table_args__ = (UniqueConstraint("user_id", "provider", name="uq_api_keys_user_provider"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)  # openai, deepseek
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
    index_signature: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
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
    __table_args__ = (
        Index("ix_index_jobs_status_created", "status", "created_at"),
    )

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
