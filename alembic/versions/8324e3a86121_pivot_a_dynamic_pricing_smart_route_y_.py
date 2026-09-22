"""pivot a DynamicPricing/SmartRoute/LeadMatch: saca alert_tickets y reduce casetype a lead_match

Los 4 casos originales (mensajería, pricing RL, anomalías, acelerador) se
reemplazaron por 3 productos nuevos (ver README.md, 2026-09-22). De los 3,
solo LeadMatch entrena algo con un dataset subido -- ver app/models/dataset.py
sobre por qué el enum queda con un solo valor.

Revision ID: 8324e3a86121
Revises: 9db23681f9ee
Create Date: 2026-09-22 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8324e3a86121"
down_revision: str | None = "9db23681f9ee"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_VALUES = ("case1_messaging", "case2_pricing", "case3_anomaly", "case4_accelerator")
NEW_VALUES = ("lead_match",)


def upgrade() -> None:
    op.drop_index(op.f("ix_alert_tickets_id"), table_name="alert_tickets")
    op.drop_table("alert_tickets")

    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        # SQLite (desarrollo/tests) no tiene un tipo ENUM nativo -- SQLAlchemy lo
        # mapea a VARCHAR con un CHECK que create_all ya arma con los valores
        # correctos en cada corrida limpia. No hace falta migrar nada acá.
        return

    # IMPORTANTE: esto borra cualquier dataset que haya quedado cargado para los
    # 4 casos viejos -- es intencional (son productos que ya no existen en la
    # plataforma), pero significa que esta migración es destructiva para datos
    # de esos casos si llegara a haber algo real cargado en producción.
    op.execute(
        sa.text("DELETE FROM datasets WHERE case IN :old_values").bindparams(
            sa.bindparam("old_values", expanding=True)
        ),
        {"old_values": list(OLD_VALUES)},
    )

    op.execute("ALTER TYPE casetype RENAME TO casetype_old")
    new_enum = sa.Enum(*NEW_VALUES, name="casetype")
    new_enum.create(bind)
    op.execute("ALTER TABLE datasets ALTER COLUMN case TYPE casetype USING case::text::casetype")
    op.execute("DROP TYPE casetype_old")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TYPE casetype RENAME TO casetype_new")
        old_enum = sa.Enum(*OLD_VALUES, name="casetype")
        old_enum.create(bind)
        op.execute("ALTER TABLE datasets ALTER COLUMN case TYPE casetype USING case::text::casetype")
        op.execute("DROP TYPE casetype_new")

    op.create_table(
        "alert_tickets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("alert_id", sa.String(length=255), nullable=False),
        sa.Column("action", sa.String(length=50), nullable=False),
        sa.Column("ticket_created", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["company_id"], ["companies.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("company_id", "alert_id", name="uq_alert_tickets_company_alert"),
    )
    op.create_index(op.f("ix_alert_tickets_id"), "alert_tickets", ["id"], unique=False)
