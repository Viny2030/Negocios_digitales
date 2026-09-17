"""agrega alert_tickets para idempotencia de case3_anomaly

Revision ID: 9db23681f9ee
Revises: a7e58d890372
Create Date: 2026-09-17 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9db23681f9ee"
down_revision: str | None = "a7e58d890372"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "alert_tickets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("alert_id", sa.String(length=255), nullable=False),
        sa.Column("action", sa.String(length=50), nullable=False),
        sa.Column("ticket_created", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["companies.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("company_id", "alert_id", name="uq_alert_tickets_company_alert"),
    )
    op.create_index(op.f("ix_alert_tickets_id"), "alert_tickets", ["id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_alert_tickets_id"), table_name="alert_tickets")
    op.drop_table("alert_tickets")
