"""Endpoints de SmartRoute Fleet."""

from fastapi import APIRouter, Depends

from app.cases.smart_route.schemas import RoutingRequest, RoutingResponse
from app.cases.smart_route.service import plan_routes
from app.models.company import Company
from security import get_current_company

router = APIRouter(prefix="/smart-route", tags=["smart-route"])


@router.post("/plan-routes", response_model=RoutingResponse)
def plan_routes_endpoint(
    payload: RoutingRequest,
    company: Company = Depends(get_current_company),  # noqa: ARG001 -- autentica/gatea el request
):
    """Dadas las paradas (con ventana horaria y demanda) y la flota disponible,
    devuelve la asignación de paradas por vehículo y el orden de la ruta."""
    return plan_routes(payload)
