"""Router raíz de la API v1: agrega endpoints comunes (empresas, datasets) y los 3 casos."""
from fastapi import APIRouter

from app.api.v1.endpoints import companies, datasets
from app.cases.case1_messaging.router import router as case1_router
from app.cases.case2_pricing.router import router as case2_router
from app.cases.case3_anomaly.router import router as case3_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(companies.router)
api_router.include_router(datasets.router)
api_router.include_router(case1_router)
api_router.include_router(case2_router)
api_router.include_router(case3_router)
