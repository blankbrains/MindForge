"""Database layer — SQLAlchemy with SQLite (swap to PostgreSQL in production)."""

from __future__ import annotations

import os
import hashlib
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    String, Text, Float, DateTime, ForeignKey,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

# ---------------------------------------------------------------------------
# Engine — PostgreSQL preferred, SQLite fallback
# ---------------------------------------------------------------------------

_DEFAULT_PG_URL = "postgresql://mindforge:mindforge@localhost:5432/mindforge"
_DEFAULT_SQLITE_URL = (
    f"sqlite:///{Path(__file__).resolve().parent.parent.parent / 'data' / 'mindforge.db'}"
)

_DB_URL = os.getenv("DATABASE_URL", "")

if not _DB_URL:
    # Try PostgreSQL first (Docker), fall back to SQLite
    try:
        import psycopg2  # noqa: F401 — probe driver availability
        _test_engine = create_engine(_DEFAULT_PG_URL, echo=False)
        _test_engine.connect().close()
        _DB_URL = _DEFAULT_PG_URL
    except Exception:
        _DB_URL = _DEFAULT_SQLITE_URL

_engine = create_engine(
    _DB_URL,
    connect_args={"check_same_thread": False} if "sqlite" in _DB_URL else {},
    echo=False,
    pool_pre_ping=True if "postgresql" in _DB_URL else False,
    pool_size=5 if "postgresql" in _DB_URL else 0,
)

SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)


def get_db() -> Session:
    """Return a new DB session. Caller must close it."""
    return SessionLocal()


# ---------------------------------------------------------------------------
# Encrypted API key helpers (simple AES-like XOR with app secret, NOT for PCI)
# ---------------------------------------------------------------------------

def _get_secret() -> bytes:
    secret = os.getenv("APP_SECRET", "")
    if not secret:
        secret = "mindforge-default-secret-change-in-production"
        import logging
        logging.getLogger(__name__).warning(
            "APP_SECRET not set — using default encryption key. "
            "Set APP_SECRET env var for production."
        )
    return hashlib.sha256(secret.encode()).digest()


def encrypt_api_key(plain: str) -> str:
    """Encrypt an API key for storage. Not bulletproof but better than plaintext."""
    if not plain:
        return ""
    secret = _get_secret()
    # XOR with repeating key + base64 encode
    key_bytes = plain.encode("utf-8")
    encrypted = bytes(b ^ secret[i % len(secret)] for i, b in enumerate(key_bytes))
    return encrypted.hex()


def decrypt_api_key(encrypted: str) -> str:
    """Decrypt a stored API key."""
    if not encrypted:
        return ""
    secret = _get_secret()
    try:
        raw = bytes.fromhex(encrypted)
        decrypted = bytes(b ^ secret[i % len(secret)] for i, b in enumerate(raw))
        return decrypted.decode("utf-8")
    except Exception:
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
        return salt + ":" + hashlib.sha256((salt + password).encode()).hexdigest()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        try:
            salt, h = hashed.split(":", 1)
            return h == hashlib.sha256((salt + password).encode()).hexdigest()
        except Exception:
            return False


class ApiKey(Base):
    __tablename__ = "api_keys"

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


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=_engine)

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
