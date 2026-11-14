"""guarda el hash de la API key en vez de la key en claro

Las keys existentes siguen funcionando: se calcula el hash de cada una y después se
borra la columna en claro.

Revision ID: c41f7a2e9b10
Revises: 8324e3a86121
Create Date: 2026-11-09 00:00:00.000000

"""

import hashlib
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c41f7a2e9b10"
down_revision: str | None = "8324e3a86121"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

API_KEY_PREFIX_LENGTH = 11


def upgrade() -> None:
    with op.batch_alter_table("companies") as batch:
        batch.add_column(sa.Column("api_key_hash", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("api_key_prefix", sa.String(length=16), nullable=True))

    companies = sa.table(
        "companies",
        sa.column("id", sa.Integer),
        sa.column("api_key", sa.String),
        sa.column("api_key_hash", sa.String),
        sa.column("api_key_prefix", sa.String),
    )
    bind = op.get_bind()
    for row in bind.execute(sa.select(companies.c.id, companies.c.api_key)).all():
        bind.execute(
            companies.update()
            .where(companies.c.id == row.id)
            .values(
                api_key_hash=hashlib.sha256(row.api_key.encode("utf-8")).hexdigest(),
                api_key_prefix=row.api_key[:API_KEY_PREFIX_LENGTH],
            )
        )

    with op.batch_alter_table("companies") as batch:
        batch.drop_index(op.f("ix_companies_api_key"))
        batch.drop_column("api_key")
        batch.alter_column("api_key_hash", existing_type=sa.String(length=64), nullable=False)
        batch.alter_column("api_key_prefix", existing_type=sa.String(length=16), nullable=False)
        batch.create_index(op.f("ix_companies_api_key_hash"), ["api_key_hash"], unique=True)


def downgrade() -> None:
    raise NotImplementedError(
        "No se puede volver atrás: las API keys en claro no se pueden recuperar a partir "
        "de su hash. Restaurá un backup de la base si necesitás el esquema anterior."
    )
