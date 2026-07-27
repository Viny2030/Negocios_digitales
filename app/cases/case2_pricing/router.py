"""Endpoints del Caso 2.1 — Pricing dinámico con restricciones de equidad."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_company
from app.models.company import Company
from app.cases.case2_pricing.schemas import PriceRequest, PriceResponse, TrainRequest
from app.cases.case2_pricing.service import get_price

router = APIRouter(prefix="/case2", tags=["case2 - dynamic pricing"])


@router.post("/get-price", response_model=PriceResponse)
def get_price_endpoint(
    payload: PriceRequest,
    company: Company = Depends(get_current_company),
):
    """Endpoint /api/v1/case2/get-price: responde el precio final en <50ms objetivo."""
    return get_price(company.id, payload)


@router.post("/train", status_code=202)
def train_endpoint(
    payload: TrainRequest,
    db: Session = Depends(get_db),
    company: Company = Depends(get_current_company),
):
    """Encola el (re)entrenamiento en un worker de Celery (ver app/tasks.py). La versión
    resultante solo se promueve a producción si pasa el filtro de equidad (Demographic
    Parity, ver app/cases/case2_pricing/fairness.py); si no pasa, la tarea igual termina
    con éxito pero el modelo queda sin promover — chequear el campo "approved" del resultado
    de la tarea (o el historial en model_registry.list_versions)."""
    from app.tasks import train_case2_task

    async_result = train_case2_task.delay(
        company.id, payload.dataset_id, payload.algorithm,
        payload.timesteps, payload.fairness_penalty_weight,
    )
    return {
        "status": "training_started", "task_id": async_result.id,
        "company_id": company.id, "dataset_id": payload.dataset_id,
    }
