"""Endpoints del Caso 1.1 — Secuenciación de mensajes multicanal."""
from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_company
from app.models.company import Company
from app.cases.case1_messaging.schemas import RecommendationRequest, RecommendationResponse, TrainRequest
from app.cases.case1_messaging.service import invalidate_policy_cache, recommend_next_touchpoint

router = APIRouter(prefix="/case1", tags=["case1 - messaging sequencing"])


@router.post("/recommend-next-touchpoint", response_model=RecommendationResponse)
def recommend_next_touchpoint_endpoint(
    payload: RecommendationRequest,
    company: Company = Depends(get_current_company),
):
    """Dado el perfil de interacción de un prospecto, devuelve el próximo touchpoint óptimo."""
    return recommend_next_touchpoint(company.id, payload.prospect)


@router.post("/train", status_code=202)
def train_endpoint(
    payload: TrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    company: Company = Depends(get_current_company),
):
    """Dispara el (re)entrenamiento del modelo de la empresa en background.
    Implementación real en scripts/train_case1.py (se invoca aquí como subproceso o Celery task)."""
    from scripts.train_case1 import train_for_company  # import diferido para no cargar SB3 en cada request

    background_tasks.add_task(train_for_company, company.id, payload.dataset_id, payload.algorithm, payload.timesteps)
    background_tasks.add_task(invalidate_policy_cache, company.id)
    return {"status": "training_started", "company_id": company.id, "dataset_id": payload.dataset_id}
