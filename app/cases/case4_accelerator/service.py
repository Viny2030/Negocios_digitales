"""Lógica de negocio del Caso 6.2: estima el valor esperado de cada startup del
portfolio y resuelve la asignación de recursos del período vía LP."""
import os

from app.core.config import get_settings
from app.cases.case4_accelerator.constraints import apply_portfolio_allocation
from app.cases.case4_accelerator.policy import AcceleratorPolicy
from app.cases.case4_accelerator.schemas import AllocationRequest, AllocationResponse

settings = get_settings()

_policy_cache: dict[int, AcceleratorPolicy] = {}


def _get_policy_for_company(company_id: int) -> AcceleratorPolicy:
    if company_id not in _policy_cache:
        model_path = os.path.join(settings.MODELS_DIR, str(company_id), "case4_accelerator", "model.zip")
        _policy_cache[company_id] = AcceleratorPolicy(model_path=model_path)
    return _policy_cache[company_id]


def allocate_resources(company_id: int, request: AllocationRequest) -> AllocationResponse:
    policy = _get_policy_for_company(company_id)
    value_scores = policy.score_portfolio(request.startups, request.resources)

    allocations, constraints_applied = apply_portfolio_allocation(
        value_scores, request.startups, request.resources,
    )

    if allocations:
        top = max(allocations, key=lambda a: a.expected_value_score)
        rationale = (
            f"Portfolio de {len(request.startups)} startups evaluado; mayor score de valor "
            f"esperado: {top.startup_id} ({top.expected_value_score})."
        )
    else:
        rationale = "Portfolio vacío: no hay startups para asignar recursos."

    return AllocationResponse(allocations=allocations, constraints_applied=constraints_applied, rationale=rationale)


def invalidate_policy_cache(company_id: int) -> None:
    _policy_cache.pop(company_id, None)