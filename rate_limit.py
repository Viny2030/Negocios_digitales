"""
Rate limiting en tres capas:

1. Techo por IP sobre todo el tráfico (`GlobalRateLimitMiddleware`). Hace falta porque el
   límite por empresa no sabe si la key es válida: inventar una distinta en cada pedido
   arrancaría siempre con el contador en cero.
2. Límite por defecto (`GlobalRateLimitMiddleware`), por empresa: la clave es un hash del
   header X-API-Key, así dos empresas nunca se pisan el límite entre sí. Sin header, cae a
   la IP de origen. Se usa un hash porque el limitador puede escribir la clave en el log.
3. Límites más estrictos en endpoints puntuales (entrenamiento, carga de datasets, alta de
   empresas), con el decorador @limiter.limit(...) en su propio router.

Las capas 1 y 2 no usan SlowAPIMiddleware a propósito: desde FastAPI 0.162 esa clase no
encuentra las rutas agregadas con include_router, y sin avisar deja de aplicar los
límites por defecto. El decorador de la capa 3 no depende de eso y sigue funcionando.

Nota: el backend de conteo por defecto es en memoria (por proceso). Con un solo proceso de
uvicorn alcanza; si en algún momento se corre con varios workers detrás de un balanceador,
hay que apuntar `storage_uri` a un backend compartido (ej. Redis, `storage_uri="redis://..."`)
para que el conteo sea común a todos los procesos.
"""

import hashlib

from limits import parse
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from config import get_settings

settings = get_settings()


def client_ip(request: Request) -> str:
    """IP del cliente. Detrás de proxies se cuenta desde la derecha de X-Forwarded-For
    (ver TRUSTED_PROXY_HOPS): lo que está a la izquierda lo puede inventar el cliente."""
    hops = settings.TRUSTED_PROXY_HOPS
    if hops > 0:
        forwarded = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(forwarded) >= hops:
            return forwarded[-hops]
    return get_remote_address(request)


def rate_limit_key(request: Request) -> str:
    api_key = request.headers.get(settings.API_KEY_HEADER)
    if api_key:
        return "key:" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:32]
    return client_ip(request)


limiter = Limiter(key_func=rate_limit_key)


class GlobalRateLimitMiddleware:
    """Aplica el techo por IP y el límite por defecto a todos los pedidos."""

    def __init__(self, app: ASGIApp, ip_limit: str | None = None, default_limit: str | None = None) -> None:
        self.app = app
        self.ip_limit = parse(ip_limit or settings.IP_RATE_LIMIT)
        self.default_limit = parse(default_limit or settings.DEFAULT_RATE_LIMIT)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not limiter.enabled:
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        within_limits = limiter.limiter.hit(self.ip_limit, "ip-global", client_ip(request)) and (
            limiter.limiter.hit(self.default_limit, "default", rate_limit_key(request))
        )
        if not within_limits:
            response = JSONResponse(
                {"detail": "Se superó el límite de pedidos"}, status_code=429, headers={"Retry-After": "60"}
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
