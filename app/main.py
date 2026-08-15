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
        "Plataforma de algoritmos prescriptivos (RL + Optimización) para negocios digitales. "
        "Expone 4 casos: secuenciación de mensajes (1.1), pricing dinámico con equidad (2.1), "
        "triaje de anomalías (3.3) y asignación de recursos en aceleradora (6.2). Multi-empresa: "
        "cada empresa carga sus propios datasets y entrena/consulta su propio modelo."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
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
