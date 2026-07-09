"""Lógica de negocio del Caso 1.1: resuelve el modelo de la empresa y arma la respuesta."""
import os

from app.core.config import get_settings
from app.cases.case1_messaging.policy import MessagingPolicy
from app.cases.case1_messaging.schemas import ProspectProfile, RecommendationResponse

settings = get_settings()

_policy_cache: dict[int, MessagingPolicy] = {}


def _get_policy_for_company(company_id: int) -> MessagingPolicy:
    if company_id not in _policy_cache:
        model_path = os.path.join(settings.MODELS_DIR, str(company_id), "case1_messaging", "model.zip")
        _policy_cache[company_id] = MessagingPolicy(model_path=model_path)
    return _policy_cache[company_id]


def recommend_next_touchpoint(company_id: int, profile: ProspectProfile) -> RecommendationResponse:
    policy = _get_policy_for_company(company_id)
    action, value, rationale = policy.predict(profile)
    return RecommendationResponse(
        prospect_id=profile.prospect_id,
        recommended_action=action,
        expected_value=value,
        rationale=rationale,
    )


def invalidate_policy_cache(company_id: int) -> None:
    """Llamar después de reentrenar, para forzar la recarga del modelo actualizado."""
    _policy_cache.pop(company_id, None)
