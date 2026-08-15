"""
env.py de Alembic, cableado a la configuración real de la app en vez de a los
placeholders que genera `alembic init`:
  - la URL de conexión sale de settings.DATABASE_URL (mismo .env que usa la API), no de
    un valor hardcodeado en alembic.ini -- así "alembic upgrade head" siempre migra la
    misma base que usa `uvicorn app.main:app`.
  - target_metadata es Base.metadata, después de importar app.models (igual que hace
    database.init_db()) para que los dos modelos (Company, Dataset) queden registrados
    antes de que autogenerate los compare contra la base.
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# La raíz del proyecto (donde viven config.py, database.py, app/) no está en sys.path
# por defecto cuando Alembic se invoca como script -- se agrega a mano.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import company, dataset  # noqa: E402,F401  (registran las tablas en Base)
from config import get_settings  # noqa: E402
from database import Base  # noqa: E402

config = context.config
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Genera el SQL de la migración sin necesitar una conexión real a la base
    (útil para revisar el script antes de aplicarlo, ej. `alembic upgrade head --sql`)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Aplica la migración contra una conexión real (el caso normal: `alembic upgrade head`)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
