"""Endpoints del Caso 2.1 — Pricing dinámico con restricciones de equidad."""
from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_company
from app.models.company import Company
from app.cases.case2_pricing.schemas import PriceRequest, PriceResponse, TrainRequest
from app.cases.case2_pricing.service import get_price, invalidate_policy_cache

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
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    company: Company = Depends(get_current_company),
):
    from scripts.train_case2 import train_for_company

    background_tasks.add_task(
        train_for_company, company.id, payload.dataset_id, payload.algorithm,
        payload.timesteps, payload.fairness_penalty_weight,
    )
    background_tasks.add_task(invalidate_policy_cache, company.id)
    return {"status": "training_started", "company_id": company.id, "dataset_id": payload.dataset_id}
