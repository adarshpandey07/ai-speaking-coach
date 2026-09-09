"""Postgres engine + session helpers.

SQLAlchemy is used synchronously. FastAPI's WebSocket handler is async, so
every call into here from async code goes through
`starlette.concurrency.run_in_threadpool` — see app/main.py. That keeps the
event loop free without dragging an async driver into the MVP.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.models import Base

log = logging.getLogger(__name__)


class DatabaseUnavailable(RuntimeError):
    """Raised with an actionable message instead of a raw driver traceback."""


def _sqlalchemy_url(raw: str) -> str:
    # psycopg3 is what we install; without the explicit driver SQLAlchemy
    # reaches for psycopg2 and fails with a confusing ImportError.
    if raw.startswith("postgresql://"):
        return raw.replace("postgresql://", "postgresql+psycopg://", 1)
    return raw


_settings = get_settings()
engine = create_engine(
    _sqlalchemy_url(_settings.database_url),
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=5,
    future=True,
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def db_session() -> Iterator[OrmSession]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection() -> None:
    """Fail fast, and fail readably, when Postgres is not reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("select 1"))
    except SQLAlchemyError as exc:  # pragma: no cover - environment dependent
        raise DatabaseUnavailable(
            "Cannot reach Postgres at "
            f"{_settings.database_url}.\n"
            "Start it with:\n"
            "  docker run -d --name speaking-coach-pg "
            "-e POSTGRES_USER=coach -e POSTGRES_PASSWORD=coach "
            "-e POSTGRES_DB=speaking_coach -p 5432:5432 postgres:16-alpine\n"
            f"Driver said: {exc.__class__.__name__}: {exc}"
        ) from exc


def create_tables() -> None:
    Base.metadata.create_all(engine)
    log.info("tables ready")
