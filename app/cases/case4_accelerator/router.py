"""Endpoints del Caso 6.2 — Asignación de recursos en incubadora/aceleradora."""
from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_company
from app.models.company import Company
from app.cases.case4_accelerator.schemas import AllocationRequest, AllocationResponse, TrainRequest
from app.cases.case4_accelerator.service import allocate_resources, invalidate_policy_cache

router = APIRouter(prefix="/case4", tags=["case4 - accelerator resource allocation"])


@router.post("/allocate", response_model=AllocationResponse)
def allocate_resources_endpoint(
    payload: AllocationRequest,
    company: Company = Depends(get_current_company),
):
    """Dado el portfolio de startups y los recursos disponibles del período, devuelve
    la asignación óptima de mentoría, capital follow-on y conexiones estratégicas."""
    return allocate_resources(company.id, payload)


@router.post("/train", status_code=202)
def train_endpoint(
    payload: TrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    company: Company = Depends(get_current_company),
):
    """Dispara el (re)entrenamiento del modelo de la empresa en background.
    Implementación real en scripts/train_case4.py."""
    from scripts.train_case4 import train_for_company

    background_tasks.add_task(train_for_company, company.id, payload.dataset_id, payload.algorithm, payload.timesteps)
    background_tasks.add_task(invalidate_policy_cache, company.id)
    return {"status": "training_started", "company_id": company.id, "dataset_id": payload.dataset_id}