"""Lógica de negocio del Caso 2.1: infiere el precio RL y aplica el filtro de factibilidad."""
import time

import model_registry
from config import get_settings
from app.cases.case2_pricing.constraints import apply_feasibility_filter
from app.cases.case2_pricing.policy import PricingPolicy
from app.cases.case2_pricing.schemas import PriceRequest, PriceResponse

settings = get_settings()
CASE = "case2_pricing"

# Ver comentario equivalente en case1_messaging/service.py: cachear junto al version_id
# permite detectar un reentrenamiento (aprobado por el filtro de equidad) corrido en otro
# proceso -- p.ej. un worker de Celery -- sin depender de invalidate_policy_cache().
_policy_cache: dict[int, tuple[str | None, PricingPolicy]] = {}


def _get_policy_for_company(company_id: int) -> PricingPolicy:
    current_version_id = model_registry.get_current_version_id(settings.MODELS_DIR, company_id, CASE)
    cached = _policy_cache.get(company_id)
    if cached is None or cached[0] != current_version_id:
        # Sirve la última versión aprobada por el registro (ver app/core/model_registry.py);
        # si ninguna fue promovida todavía (o falló el filtro de equidad), model_path queda
        # en None y PricingPolicy cae a su heurística de respaldo.
        model_path = model_registry.get_current_artifact_path(settings.MODELS_DIR, company_id, CASE, "model.zip")
        cached = (current_version_id, PricingPolicy(model_path=model_path))
        _policy_cache[company_id] = cached
    return cached[1]


def get_price(company_id: int, request: PriceRequest) -> PriceResponse:
    start = time.perf_counter()
    policy = _get_policy_for_company(company_id)

    raw_price = policy.predict_price(request)
    final_price, constraints_applied = apply_feasibility_filter(raw_price, request)

    latency_ms = (time.perf_counter() - start) * 1000
    return PriceResponse(
        product_id=request.product_id,
        final_price=final_price,
        raw_rl_price=round(raw_price, 2),
        constraints_applied=constraints_applied,
        latency_ms=round(latency_ms, 2),
    )


def invalidate_policy_cache(company_id: int) -> None:
    _policy_cache.pop(company_id, None)
