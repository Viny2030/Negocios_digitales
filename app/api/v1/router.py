"""Router raíz de la API v1: agrega endpoints comunes (empresas, datasets) y los
3 productos de la plataforma (ver README.md para el detalle de cada uno)."""

from fastapi import APIRouter

from app.api.v1.endpoints import companies, datasets
from app.cases.dynamic_pricing.router import router as dynamic_pricing_router
from app.cases.lead_match.router import router as lead_match_router
from app.cases.smart_route.router import router as smart_route_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(companies.router)
api_router.include_router(datasets.router)
api_router.include_router(dynamic_pricing_router)
api_router.include_router(smart_route_router)
api_router.include_router(lead_match_router)
