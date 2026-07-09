"""Endpoints del Caso 3.3 - Triaje y priorizacion de anomalias."""
from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_company
from app.models.company import Company
from app.cases.case3_anomaly.schemas import AlertWebhookPayload, TeamCapacity, TrainRequest, TriageDecision
from app.cases.case3_anomaly.service import handle_alert_webhook, invalidate_policy_cache

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
def train_endpoint(
    payload: TrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    company: Company = Depends(get_current_company),
):
    from scripts.train_case3 import train_for_company

    background_tasks.add_task(train_for_company, company.id, payload.dataset_id, payload.timesteps)
    background_tasks.add_task(invalidate_policy_cache, company.id)
    return {"status": "training_started", "company_id": company.id, "dataset_id": payload.dataset_id}
