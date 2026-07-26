"""
Caso 1.1 — Política óptima de secuenciación de mensajes en campañas multicanal.

Estado (S): perfil de interacción acumulado del prospecto.
Acciones (A): touchpoints disponibles (email A/B/C, retargeting Meta, SMS, push, llamada, esperar).
Recompensa (R): conversión neta del costo de contacto, penalización por fatiga.
"""
import datetime as dt
import enum

from pydantic import BaseModel, Field


class Touchpoint(str, enum.Enum):
    email_a = "email_a"
    email_b = "email_b"
    email_c = "email_c"
    retargeting_meta = "retargeting_meta"
    sms = "sms"
    push = "push"
    call = "call"
    wait = "wait"  # esperar N días, no contactar todavía


class ProspectProfile(BaseModel):
    """Estado (S) del prospecto: se construye a partir del dataset cargado por la empresa
    o se recibe directamente en el request si la empresa ya lo calcula en su CRM."""

    prospect_id: str
    channels_contacted: list[str] = Field(default_factory=list)
    messages_received: int = 0
    responses_count: int = 0
    days_since_last_contact: float = 0.0
    purchase_stage: str = "awareness"  # awareness | consideration | decision
    unsubscribed: bool = False

    # --- Campos usados por el filtro de restricciones (constraints.py) ---
    days_since_last_email: float | None = Field(
        default=None,
        description="Días desde el último email enviado (cualquier variante A/B/C). "
        "None si nunca se le envió un email a este prospecto.",
    )
    contacts_this_week: dict[str, int] = Field(
        default_factory=dict,
        description="Contactos ya realizados esta semana por bucket de canal "
        "(email/sms/push/call/retargeting_meta), para aplicar el presupuesto semanal.",
    )


class ChannelBudget(BaseModel):
    """Presupuesto semanal por canal para un prospecto (tope de contactos permitidos).
    Default de negocio: 1 email por semana, cualquier variante A/B/C."""

    email: int = Field(default=1, ge=0)
    sms: int = Field(default=2, ge=0)
    push: int = Field(default=3, ge=0)
    call: int = Field(default=1, ge=0)
    retargeting_meta: int = Field(default=5, ge=0)


class RecommendationRequest(BaseModel):
    prospect: ProspectProfile
    channel_budget: ChannelBudget = Field(default_factory=ChannelBudget)


class RecommendationResponse(BaseModel):
    prospect_id: str
    recommended_action: Touchpoint
    expected_value: float
    rationale: str
    constraints_applied: list[str] = Field(default_factory=list)


class TrainRequest(BaseModel):
    dataset_id: int
    algorithm: str = Field(default="dqn", description="dqn | ppo")
    timesteps: int = 50_000