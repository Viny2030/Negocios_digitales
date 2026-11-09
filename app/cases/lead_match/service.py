"""Orquesta LeadMatch: resuelve el scorer vigente de la empresa (o la
heurística de respaldo), puntúa cada lead y resuelve el matching ponderado."""

import model_registry
from app.cases.lead_match.matching import match_leads
from app.cases.lead_match.schemas import MatchRequest, MatchResponse
from app.cases.lead_match.scoring import ConversionScorer, load_scorer
from config import get_settings

settings = get_settings()
CASE = "lead_match"

_scorer_cache: dict[int, tuple[str | None, ConversionScorer]] = {}


def _get_scorer_for_company(company_id: int) -> ConversionScorer:
    current_version_id = model_registry.get_current_version_id(settings.MODELS_DIR, company_id, CASE)
    cached = _scorer_cache.get(company_id)
    if cached is None or cached[0] != current_version_id:
        model_path = model_registry.get_current_artifact_path(
            settings.MODELS_DIR, company_id, CASE, "model.joblib"
        )
        cached = (current_version_id, load_scorer(model_path))
        _scorer_cache[company_id] = cached
    return cached[1]


def invalidate_scorer_cache(company_id: int) -> None:
    _scorer_cache.pop(company_id, None)


def match(company_id: int, request: MatchRequest) -> MatchResponse:
    scorer = _get_scorer_for_company(company_id)

    conversion_probabilities = scorer.predict_many(request.leads)

    assignments, unmatched = match_leads(
        leads=request.leads,
        conversion_probabilities=conversion_probabilities,
        reps=request.reps,
        specialty_bonus=request.specialty_bonus,
    )

    return MatchResponse(
        assignments=assignments,
        unmatched_leads=unmatched,
        used_default_scoring=scorer.is_default,
    )
