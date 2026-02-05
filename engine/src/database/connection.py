"""Database connection management for AG Analyzer."""

import os
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import structlog

from .models import Base

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(
        self,
        database_url: Optional[str] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        echo: bool = False,
    ):
        """Initialize database manager.

        Args:
            database_url: PostgreSQL connection URL
            pool_size: Connection pool size
            max_overflow: Max overflow connections
            echo: Echo SQL statements
        """
        if database_url is None:
            database_url = self._build_url_from_env()

        self._engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            echo=echo,
            pool_pre_ping=True,  # Verify connections before use
        )

        self._session_factory = sessionmaker(
            bind=self._engine,
            autocommit=False,
            autoflush=False,
        )

        # Log connection events
        @event.listens_for(self._engine, "connect")
        def on_connect(dbapi_conn, connection_record):
            logger.debug("Database connection established")

        @event.listens_for(self._engine, "checkout")
        def on_checkout(dbapi_conn, connection_record, connection_proxy):
            logger.debug("Connection checked out from pool")

    @staticmethod
    def _build_url_from_env() -> str:
        """Build database URL from environment variables."""
        db_name = os.getenv("DB_NAME", "ag_analyzer")
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "postgres")
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")

        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    def create_tables(self) -> None:
        """Create all tables in the database."""
        logger.info("Creating database tables")
        Base.metadata.create_all(self._engine)

    def drop_tables(self) -> None:
        """Drop all tables in the database."""
        logger.warning("Dropping all database tables")
        Base.metadata.drop_all(self._engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Get a database session context manager.

        Usage:
            with db.session() as session:
                session.query(Model).all()
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Database session error", error=str(e))
            raise
        finally:
            session.close()

    def get_session(self) -> Session:
        """Get a new database session (caller must manage lifecycle)."""
        return self._session_factory()

    @property
    def engine(self):
        """Get the SQLAlchemy engine."""
        return self._engine

    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            with self.session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def init_db(database_url: Optional[str] = None, **kwargs) -> DatabaseManager:
    """Initialize the global database manager.

    Args:
        database_url: PostgreSQL connection URL
        **kwargs: Additional arguments for DatabaseManager

    Returns:
        DatabaseManager instance
    """
    global _db_manager
    _db_manager = DatabaseManager(database_url, **kwargs)
    return _db_manager


def get_db() -> DatabaseManager:
    """Get the global database manager.

    Returns:
        DatabaseManager instance

    Raises:
        RuntimeError: If database not initialized
    """
    if _db_manager is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _db_manager


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Convenience function to get a database session.

    Usage:
        with get_session() as session:
            session.query(Model).all()
    """
    db = get_db()
    with db.session() as session:
        yield session
