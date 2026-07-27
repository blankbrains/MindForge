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
    String, Text, Float, DateTime, ForeignKey, UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from cryptography.fernet import Fernet, InvalidToken

# ---------------------------------------------------------------------------
# Engine — PostgreSQL only
# ---------------------------------------------------------------------------

_DEFAULT_PG_URL = "postgresql://mindforge:mindforge@localhost:5432/mindforge"
_DB_URL = os.getenv("DATABASE_URL", _DEFAULT_PG_URL)

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
