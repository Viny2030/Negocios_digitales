"""Endpoints de DynamicPricing API."""

from fastapi import APIRouter, Depends

from app.cases.dynamic_pricing.schemas import PricingRequest, PricingResponse
from app.cases.dynamic_pricing.service import price_skus
from app.models.company import Company
from security import get_current_company

router = APIRouter(prefix="/dynamic-pricing", tags=["dynamic-pricing"])


@router.post("/price", response_model=PricingResponse)
def price_endpoint(
    payload: PricingRequest,
    company: Company = Depends(get_current_company),  # noqa: ARG001 -- autentica/gatea el request
):
    """Dado el estado de cada SKU (costo, stock, precio actual, eventos de compra
    recientes) devuelve el precio recomendado en tiempo real, respetando el margen
    mínimo y la variación diaria máxima."""
    return price_skus(payload)
