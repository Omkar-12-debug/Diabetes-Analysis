"""SQLAlchemy session management, engine configuration, and SQLite fallback logic.

Supports PostgreSQL via DATABASE_URL with seamless local SQLite fallback:
sqlite:///./data/diabetes_app.db
"""

import logging
import os
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SQLITE_URL = "sqlite:///./data/diabetes_app.db"
ENV_DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE_URL)


def _ensure_sqlite_directory(url: str) -> None:
    """Ensure parent directory exists for local SQLite database file."""
    if url.startswith("sqlite:///"):
        db_path_str = url.replace("sqlite:///", "")
        db_path = Path(db_path_str)
        if db_path.parent and not db_path.parent.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)


def create_db_engine(database_url: str):
    """Create SQLAlchemy engine with connection test and SQLite fallback.

    Args:
        database_url: Connection URL string.

    Returns:
        Configured SQLAlchemy Engine.
    """
    _ensure_sqlite_directory(database_url)
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}

    try:
        engine = create_engine(database_url, connect_args=connect_args)
        # Verify connection
        with engine.connect():
            pass
        logger.info("Successfully connected to primary database at: %s", database_url)
        return engine
    except Exception as exc:
        if database_url != DEFAULT_SQLITE_URL:
            logger.warning(
                "Primary database (%s) unreachable: %s. Falling back to local SQLite at %s",
                database_url,
                exc,
                DEFAULT_SQLITE_URL,
            )
            _ensure_sqlite_directory(DEFAULT_SQLITE_URL)
            return create_engine(DEFAULT_SQLITE_URL, connect_args={"check_same_thread": False})
        raise


engine = create_db_engine(ENV_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initialize database tables from ORM metadata."""
    logger.info("Initializing database tables...")
    _ensure_sqlite_directory(str(engine.url))
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a scoped database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
