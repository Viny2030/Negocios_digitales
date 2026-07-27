"""Endpoints del Caso 1.1 — Secuenciación de mensajes multicanal."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_company
from app.models.company import Company
from app.cases.case1_messaging.schemas import RecommendationRequest, RecommendationResponse, TrainRequest
from app.cases.case1_messaging.service import recommend_next_touchpoint

router = APIRouter(prefix="/case1", tags=["case1 - messaging sequencing"])


@router.post("/recommend-next-touchpoint", response_model=RecommendationResponse)
def recommend_next_touchpoint_endpoint(
    payload: RecommendationRequest,
    company: Company = Depends(get_current_company),
):
    """Dado el perfil de interacción de un prospecto, devuelve el próximo touchpoint óptimo
    ya filtrado por las restricciones de cadencia (1 email/semana) y presupuesto por canal."""
    return recommend_next_touchpoint(company.id, payload.prospect, payload.channel_budget)


@router.post("/train", status_code=202)
def train_endpoint(
    payload: TrainRequest,
    db: Session = Depends(get_db),
    company: Company = Depends(get_current_company),
):
    """Encola el (re)entrenamiento del modelo de la empresa en un worker de Celery
    (ver app/tasks.py). Requiere un broker Redis corriendo y al menos un worker levantado."""
    from app.tasks import train_case1_task

    async_result = train_case1_task.delay(company.id, payload.dataset_id, payload.algorithm, payload.timesteps)
    return {
        "status": "training_started", "task_id": async_result.id,
        "company_id": company.id, "dataset_id": payload.dataset_id,
    }
