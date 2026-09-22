"""
Punto de entrada de la aplicación FastAPI.
Ejecutar en desarrollo con:  uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

import logging_config  # noqa: F401  (configura logging por su efecto secundario, ver el módulo)
from app.api.v1.router import api_router
from config import get_settings
from database import init_db
from rate_limit import limiter

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Suite de APIs B2B de optimización y ML para negocios digitales: DynamicPricing "
        "(fijación dinámica de precios con restricciones de margen/variación), SmartRoute "
        "Fleet (ruteo y despacho last-mile) y LeadMatch Engine (scoring y asignación de "
        "leads a vendedores). Multi-empresa: cada empresa cliente tiene su propio API key, "
        "aislado del resto."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# `allow_origins=["*"]` (el default, ver settings.CORS_ORIGINS) + `allow_credentials=True`
# es una combinación inválida según la spec de CORS (un origen comodín no puede llevar
# credenciales) -- Starlette la "resuelve" reflejando el Origin real de cada request en
# vez de "*", lo que en la práctica abre credenciales a cualquier origen. Como la auth acá
# es un header explícito (X-API-Key), no cookies/sesión, no hace falta allow_credentials
# salvo que se configuren orígenes específicos (no "*").
_cors_allow_all_origins = "*" in settings.CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=not _cors_allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting por empresa (ver rate_limit.py). El límite global (120/minuto por API key)
# se aplica acá vía middleware; algunos endpoints pesados tienen además un límite más
# estricto explícito en su propio router (ej. /train, carga de datasets).
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "project": settings.PROJECT_NAME}


app.include_router(api_router)
