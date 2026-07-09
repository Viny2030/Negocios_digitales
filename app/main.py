"""
Punto de entrada de la aplicación FastAPI.
Ejecutar en desarrollo con:  uvicorn app.main:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.database import init_db

settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Plataforma de algoritmos prescriptivos (RL + Optimización) para negocios digitales. "
        "Expone 3 casos: secuenciación de mensajes (1.1), pricing dinámico con equidad (2.1) "
        "y triaje de anomalías (3.3). Multi-empresa: cada empresa carga sus propios datasets "
        "y entrena/consulta su propio modelo."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "project": settings.PROJECT_NAME}


app.include_router(api_router)
