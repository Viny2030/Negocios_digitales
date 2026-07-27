"""Endpoints del Caso 6.2 — Asignación de recursos en incubadora/aceleradora."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_company
from app.models.company import Company
from app.cases.case4_accelerator.schemas import AllocationRequest, AllocationResponse, TrainRequest
from app.cases.case4_accelerator.service import allocate_resources

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
    db: Session = Depends(get_db),
    company: Company = Depends(get_current_company),
):
    """Encola el (re)entrenamiento en un worker de Celery (ver app/tasks.py)."""
    from app.tasks import train_case4_task

    async_result = train_case4_task.delay(company.id, payload.dataset_id, payload.algorithm, payload.timesteps)
    return {
        "status": "training_started", "task_id": async_result.id,
        "company_id": company.id, "dataset_id": payload.dataset_id,
    }