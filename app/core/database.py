"""
Configuración de base de datos (SQLAlchemy).
Multi-tenant simple: todas las tablas llevan company_id como foreign key,
en vez de una base de datos por empresa (más fácil de operar en el MVP).
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import get_settings

settings = get_settings()

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency de FastAPI para inyectar una sesión de DB por request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Crea las tablas si no existen. Para producción, usar Alembic migrations."""
    from app.models import company, dataset  # noqa: F401  (registran los modelos en Base)

    Base.metadata.create_all(bind=engine)
