"""Endpoints de LeadMatch Engine."""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.cases.lead_match.schemas import MatchRequest, MatchResponse, TrainRequest, TrainResponse
from app.cases.lead_match.service import invalidate_scorer_cache, match
from app.cases.lead_match.train import InsufficientTrainingDataError, train_for_company
from app.models.company import Company
from app.models.dataset import CaseType, Dataset
from database import get_db
from rate_limit import limiter
from security import get_current_company

router = APIRouter(prefix="/lead-match", tags=["lead-match"])


@router.post("/match", response_model=MatchResponse)
def match_endpoint(
    payload: MatchRequest,
    company: Company = Depends(get_current_company),
):
    """Puntúa cada lead (probabilidad de conversión) y lo asigna al vendedor que
    maximiza el score esperado, respetando la capacidad de cada uno."""
    return match(company.id, payload)


@router.post("/train", response_model=TrainResponse)
@limiter.limit("5/minute")
def train_endpoint(
    request: Request,
    payload: TrainRequest,
    db: Session = Depends(get_db),
    company: Company = Depends(get_current_company),
):
    """Entrena (o reentrena) el clasificador de conversión de la empresa con un
    dataset ya subido (ver POST /datasets/lead_match). Corre sincrónico: una
    regresión logística sobre unos miles de leads tarda segundos."""
    dataset = (
        db.query(Dataset)
        .filter(
            Dataset.id == payload.dataset_id,
            Dataset.company_id == company.id,
            Dataset.case == CaseType.lead_match,
        )
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    try:
        version_id, metrics = train_for_company(company.id, dataset.file_path)
    except InsufficientTrainingDataError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    invalidate_scorer_cache(company.id)
    return TrainResponse(status="trained", version_id=version_id, metrics=metrics)
