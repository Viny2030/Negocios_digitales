"""
Rate limiting por empresa (no por IP): la clave del limitador es el header X-API-Key
cuando está presente -- identifica a la empresa cliente, así que dos empresas nunca se
pisan el límite entre sí -- y cae a la IP de origen solo para requests sin autenticar
(ej. /health o el propio POST /api/v1/companies para registrarse).

Usa slowapi (wrapper de la librería `limits` para Starlette/FastAPI). El límite por
defecto (ver `default_limits` abajo) se aplica de forma global a través de
SlowAPIMiddleware (ver app/main.py); algunos endpoints puntualmente pesados
(entrenamiento, carga de datasets) tienen además un límite explícito más estricto en
su propio router, vía el decorador @limiter.limit(...).

Nota: el backend de conteo por defecto de slowapi es en memoria (por proceso). Con un
solo proceso de uvicorn alcanza; si en algún momento se corre con varios workers detrás
de un balanceador, hay que apuntar `storage_uri` a Redis (que este proyecto ya usa para
Celery, ver settings.REDIS_URL) para que el conteo sea compartido entre procesos.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from config import get_settings

settings = get_settings()


def rate_limit_key(request: Request) -> str:
    api_key = request.headers.get(settings.API_KEY_HEADER)
    return api_key or get_remote_address(request)


limiter = Limiter(key_func=rate_limit_key, default_limits=["120/minute"])
