"""Endpoints de LeadMatch Engine."""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

import model_registry
from app.cases.lead_match.schemas import (
    MatchRequest,
    MatchResponse,
    ModelVersion,
    TrainRequest,
    TrainResponse,
)
from app.cases.lead_match.service import invalidate_scorer_cache, match
from app.cases.lead_match.train import CASE, InsufficientTrainingDataError, train_for_company
from app.models.company import Company
from app.models.dataset import CaseType, Dataset
from config import get_settings
from database import get_db
from rate_limit import limiter
from security import get_current_company

router = APIRouter(prefix="/lead-match", tags=["lead-match"])
settings = get_settings()


def _list_versions(company_id: int) -> list[ModelVersion]:
    current = model_registry.get_current_version_id(settings.MODELS_DIR, company_id, CASE)
    return [
        ModelVersion(
            version_id=version["version_id"],
            created_at=version["created_at"],
            metrics=version["metrics"],
            is_current=version["version_id"] == current,
        )
        for version in model_registry.list_versions(settings.MODELS_DIR, company_id, CASE)
    ]


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
    regresión logística sobre unos miles de leads tarda segundos. La versión nueva
    queda vigente solo si es útil y no es peor que la actual: ver `promoted`."""
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
        version_id, metrics, promoted, reason = train_for_company(company.id, dataset.file_path, dataset.id)
    except InsufficientTrainingDataError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    invalidate_scorer_cache(company.id)
    return TrainResponse(
        status="trained",
        version_id=version_id,
        metrics=metrics,
        promoted=promoted,
        promotion_reason=reason,
        current_version_id=model_registry.get_current_version_id(settings.MODELS_DIR, company.id, CASE),
    )


@router.get("/models", response_model=list[ModelVersion])
def list_models(company: Company = Depends(get_current_company)):
    """Lista las versiones entrenadas de la empresa, de la más vieja a la más nueva."""
    return _list_versions(company.id)


@router.post("/models/{version_id}/promote", response_model=ModelVersion)
def promote_model(version_id: str, company: Company = Depends(get_current_company)):
    """Deja vigente una versión ya entrenada. Sirve para volver a una anterior o para
    usar una que el entrenamiento no promovió."""
    versions = {version.version_id: version for version in _list_versions(company.id)}
    if version_id not in versions:
        raise HTTPException(status_code=404, detail="Versión no encontrada")

    model_registry.promote(settings.MODELS_DIR, company.id, CASE, version_id)
    invalidate_scorer_cache(company.id)
    return versions[version_id].model_copy(update={"is_current": True})
