"""Lógica de negocio del Caso 2.1: infiere el precio RL y aplica el filtro de factibilidad."""
import os
import time

from app.core.config import get_settings
from app.cases.case2_pricing.constraints import apply_feasibility_filter
from app.cases.case2_pricing.policy import PricingPolicy
from app.cases.case2_pricing.schemas import PriceRequest, PriceResponse

settings = get_settings()

_policy_cache: dict[int, PricingPolicy] = {}


def _get_policy_for_company(company_id: int) -> PricingPolicy:
    if company_id not in _policy_cache:
        model_path = os.path.join(settings.MODELS_DIR, str(company_id), "case2_pricing", "model.zip")
        _policy_cache[company_id] = PricingPolicy(model_path=model_path)
    return _policy_cache[company_id]


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
