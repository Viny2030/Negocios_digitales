"""
Configuración de base de datos (SQLAlchemy).
Multi-tenant simple: todas las tablas llevan company_id como foreign key,
en vez de una base de datos por empresa (más fácil de operar en el MVP).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config import get_settings

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
    """Crea las tablas si no existen. Sirve para arrancar rápido en desarrollo/tests
    (ver tests/conftest.py, que llama a Base.metadata.create_all directamente sobre su
    propia DB de test), pero no versiona cambios de esquema: si una tabla ya existe con
    una forma vieja, create_all no la altera. Para modificar el esquema sobre una base
    con datos reales, usar Alembic (ver carpeta alembic/ y el apartado "Migraciones"
    del README) en vez de tocar los modelos y confiar en que esto los actualice solo."""
    from app.models import company, dataset  # noqa: F401  (registran los modelos en Base)

    Base.metadata.create_all(bind=engine)
