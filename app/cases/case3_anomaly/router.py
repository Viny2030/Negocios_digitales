"""Endpoints del Caso 3.3 - Triaje y priorizacion de anomalias."""

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.cases.case3_anomaly.schemas import AlertWebhookPayload, TeamCapacity, TrainRequest, TriageDecision
from app.cases.case3_anomaly.service import handle_alert_webhook
from app.models.company import Company
from database import get_db
from rate_limit import limiter
from security import get_current_company

router = APIRouter(prefix="/case3", tags=["case3 - anomaly triage"])


@router.post("/alert-webhook", response_model=TriageDecision)
def alert_webhook_endpoint(
    payload: AlertWebhookPayload,
    analyst_hours_available: float = 8.0,
    pending_alerts_in_queue: int = 0,
    company: Company = Depends(get_current_company),
):
    """Endpoint /api/v1/case3/alert-webhook: recibe el payload plano que envia Datadog/Looker
    (sin envolver) y devuelve la decision de triaje priorizada. La capacidad del equipo se pasa
    como query params opcionales (en produccion, se resolveria consultando el estado interno
    de capacidad de la empresa en vez de recibirla del webhook del proveedor de monitoreo)."""
    capacity = TeamCapacity(
        analyst_hours_available=analyst_hours_available,
        pending_alerts_in_queue=pending_alerts_in_queue,
    )
    return handle_alert_webhook(company.id, payload, capacity)


@router.post("/train", status_code=202)
@limiter.limit("5/minute")
def train_endpoint(
    request: Request,
    payload: TrainRequest,
    db: Session = Depends(get_db),
    company: Company = Depends(get_current_company),
):
    """Encola el (re)entrenamiento en un worker de Celery (ver app/tasks.py)."""
    from app.tasks import train_case3_task

    async_result = train_case3_task.delay(company.id, payload.dataset_id, payload.timesteps)
    return {
        "status": "training_started",
        "task_id": async_result.id,
        "company_id": company.id,
        "dataset_id": payload.dataset_id,
    }
