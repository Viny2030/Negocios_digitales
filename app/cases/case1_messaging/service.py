"""Lógica de negocio del Caso 1.1: resuelve el modelo de la empresa, aplica las
restricciones de cadencia/presupuesto y arma la respuesta."""

import model_registry
from app.cases.case1_messaging.constraints import apply_frequency_and_budget_filter
from app.cases.case1_messaging.policy import MessagingPolicy
from app.cases.case1_messaging.schemas import ChannelBudget, ProspectProfile, RecommendationResponse
from config import get_settings

settings = get_settings()
CASE = "case1_messaging"

# Cachea (version_id, policy) por empresa. Comparar el version_id vigente en cada llamada
# (lectura barata de un JSON) permite detectar un reentrenamiento aunque haya corrido en
# otro proceso -- p.ej. un worker de Celery -- donde invalidate_policy_cache() de abajo
# nunca se ejecuta porque vive en el proceso de la API, no en el del worker.
_policy_cache: dict[int, tuple[str | None, MessagingPolicy]] = {}


def _get_policy_for_company(company_id: int) -> MessagingPolicy:
    current_version_id = model_registry.get_current_version_id(settings.MODELS_DIR, company_id, CASE)
    cached = _policy_cache.get(company_id)
    if cached is None or cached[0] != current_version_id:
        model_path = model_registry.get_current_artifact_path(settings.MODELS_DIR, company_id, CASE, "model.zip")
        metrics = model_registry.get_current_metrics(settings.MODELS_DIR, company_id, CASE) or {}
        algorithm = metrics.get("algorithm")
        cached = (current_version_id, MessagingPolicy(model_path=model_path, algorithm=algorithm))
        _policy_cache[company_id] = cached
    return cached[1]


def recommend_next_touchpoint(
    company_id: int,
    profile: ProspectProfile,
    channel_budget: ChannelBudget | None = None,
) -> RecommendationResponse:
    policy = _get_policy_for_company(company_id)
    action, value, rationale = policy.predict(profile)

    final_action, constraints_applied = apply_frequency_and_budget_filter(
        action,
        profile,
        channel_budget or ChannelBudget(),
    )
    if final_action != action:
        rationale += f" Ajustado de '{action.value}' a '{final_action.value}' por restricciones de negocio."

    return RecommendationResponse(
        prospect_id=profile.prospect_id,
        recommended_action=final_action,
        expected_value=value,
        rationale=rationale,
        constraints_applied=constraints_applied,
    )


def invalidate_policy_cache(company_id: int) -> None:
    """Llamar después de reentrenar, para forzar la recarga del modelo actualizado."""
    _policy_cache.pop(company_id, None)
