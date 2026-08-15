"""Lógica de negocio del Caso 6.2: estima el valor esperado de cada startup del
portfolio y resuelve la asignación de recursos del período vía LP."""

import model_registry
from app.cases.case4_accelerator.constraints import apply_portfolio_allocation
from app.cases.case4_accelerator.policy import AcceleratorPolicy
from app.cases.case4_accelerator.schemas import AllocationRequest, AllocationResponse
from config import get_settings

settings = get_settings()
CASE = "case4_accelerator"

# Ver comentario equivalente en case1_messaging/service.py.
_policy_cache: dict[int, tuple[str | None, AcceleratorPolicy]] = {}


def _get_policy_for_company(company_id: int) -> AcceleratorPolicy:
    current_version_id = model_registry.get_current_version_id(settings.MODELS_DIR, company_id, CASE)
    cached = _policy_cache.get(company_id)
    if cached is None or cached[0] != current_version_id:
        model_path = model_registry.get_current_artifact_path(settings.MODELS_DIR, company_id, CASE, "model.zip")
        cached = (current_version_id, AcceleratorPolicy(model_path=model_path))
        _policy_cache[company_id] = cached
    return cached[1]


def allocate_resources(company_id: int, request: AllocationRequest) -> AllocationResponse:
    policy = _get_policy_for_company(company_id)
    value_scores = policy.score_portfolio(request.startups, request.resources)

    allocations, constraints_applied = apply_portfolio_allocation(
        value_scores,
        request.startups,
        request.resources,
    )

    if allocations:
        top = max(allocations, key=lambda a: a.expected_value_score)
        rationale = (
            f"Portfolio de {len(request.startups)} startups evaluado; mayor score de valor "
            f"esperado: {top.startup_id} ({top.expected_value_score})."
        )
    else:
        rationale = "Portfolio vacío: no hay startups para asignar recursos."

    return AllocationResponse(
        allocations=allocations, constraints_applied=constraints_applied, rationale=rationale
    )


def invalidate_policy_cache(company_id: int) -> None:
    _policy_cache.pop(company_id, None)
