"""agrega content_hash, is_active y updated_at a datasets

Revision ID: a7e58d890372
Revises: 2bbd6bf8f4b4
Create Date: 2026-09-15 18:50:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7e58d890372"
down_revision: str | None = "2bbd6bf8f4b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("datasets", sa.Column("content_hash", sa.String(length=64), nullable=True))
    op.add_column(
        "datasets",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("datasets", sa.Column("updated_at", sa.DateTime(), nullable=True))
    op.create_index(op.f("ix_datasets_content_hash"), "datasets", ["content_hash"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_datasets_content_hash"), table_name="datasets")
    op.drop_column("datasets", "updated_at")
    op.drop_column("datasets", "is_active")
    op.drop_column("datasets", "content_hash")
