"""
Caso 3.3 — Política de triaje y priorización de anomalías en métricas de negocio.

Estado (S): cola de anomalías (magnitud, tiempo, contexto, historial), capacidad del equipo.
Acciones (A): ignorar, investigar superficial (5min), investigar a fondo (1h),
               escalar ingeniería, escalar negocio, fusionar con anomalía relacionada.
Recompensa (R): impacto recuperado neto del costo de investigación, penalización por
               falsos positivos investigados y por anomalías reales ignoradas.
"""
import datetime as dt
import enum

from pydantic import BaseModel, Field


class TriageAction(str, enum.Enum):
    ignore = "ignore"
    investigate_light = "investigate_light"  # 5 min
    investigate_deep = "investigate_deep"  # 1 hora
    escalate_engineering = "escalate_engineering"
    escalate_business = "escalate_business"
    merge_related = "merge_related"


class AlertWebhookPayload(BaseModel):
    """Payload recibido desde Datadog / Looker / BI vía webhook (POST /api/v1/case3/alert-webhook)."""

    alert_id: str
    metric_name: str
    z_score: float
    pct_change: float | None = None
    triggered_at: dt.datetime
    business_context: str | None = Field(default=None, description="Ej: checkout, revenue, señales de churn")
    historical_similar_count: int = 0
    estimated_financial_impact: float | None = None


class TeamCapacity(BaseModel):
    analyst_hours_available: float = 8.0
    pending_alerts_in_queue: int = 0


class TriageDecision(BaseModel):
    alert_id: str
    action: TriageAction
    priority_score: float
    false_positive_probability: float
    rationale: str
    ticket_created: bool = False


class TrainRequest(BaseModel):
    dataset_id: int
    timesteps: int = 50_000
