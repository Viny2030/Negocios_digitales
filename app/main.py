"""
Punto de entrada de la aplicación FastAPI.
Ejecutar en desarrollo con:  uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
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
    # AGREGADO 2026-09-22: se apaga el /docs default de FastAPI para reemplazarlo por
    # una versión con el mismo tema de colores azul/verde que dashboard.html (ver
    # _SWAGGER_THEME_CSS más abajo) -- puramente cosmético, no cambia el schema/OpenAPI.
    docs_url=None,
)

# Mismo criterio de color que dashboard.html: azul para GET/lectura, verde para POST/
# escritura (con un acento ámbar para PUT/DELETE, que son la minoría de los endpoints),
# sobre un fondo oscuro para que los bloques de color resalten más que en el tema
# blanco por defecto de Swagger UI.
_SWAGGER_THEME_CSS = """
<style>
  body { background: #0b1220 !important; }
  .swagger-ui { color: #e5e7eb; }
  .swagger-ui .topbar { background-color: #0b1220; border-bottom: 3px solid #2563eb; }
  .swagger-ui .topbar .download-url-wrapper { display: none; }
  .swagger-ui .info .title { color: #7dd3fc; }
  .swagger-ui .info .title small.version-stamp { background-color: #16a34a; }
  .swagger-ui .info a { color: #6ee7b7; }
  .swagger-ui .scheme-container { background: #111827; box-shadow: none; border-bottom: 1px solid #1e293b; }
  .swagger-ui .opblock-tag { color: #93c5fd; border-bottom: 1px solid #1e293b; }
  .swagger-ui .opblock-tag:hover { background: rgba(37,99,235,.08); }
  .swagger-ui .opblock.opblock-get { background: rgba(37,99,235,.08); border-color: #2563eb; }
  .swagger-ui .opblock.opblock-get .opblock-summary-method { background: #2563eb; }
  .swagger-ui .opblock.opblock-get .opblock-summary { border-color: #2563eb; }
  .swagger-ui .opblock.opblock-post { background: rgba(22,163,74,.08); border-color: #16a34a; }
  .swagger-ui .opblock.opblock-post .opblock-summary-method { background: #16a34a; }
  .swagger-ui .opblock.opblock-post .opblock-summary { border-color: #16a34a; }
  .swagger-ui .opblock.opblock-put { background: rgba(217,119,6,.08); border-color: #d97706; }
  .swagger-ui .opblock.opblock-put .opblock-summary-method { background: #d97706; }
  .swagger-ui .opblock.opblock-delete { background: rgba(220,38,38,.08); border-color: #dc2626; }
  .swagger-ui .opblock.opblock-delete .opblock-summary-method { background: #dc2626; }
  .swagger-ui .opblock .opblock-summary-description { color: #cbd5e1; }
  .swagger-ui .opblock .opblock-summary-path, .swagger-ui .opblock .opblock-summary-path__deprecated { color: #f1f5f9; }
  .swagger-ui .btn.authorize { background-color: #16a34a; border-color: #16a34a; color: #fff; }
  .swagger-ui .btn.authorize svg { fill: #fff; }
  .swagger-ui select, .swagger-ui input[type=text], .swagger-ui input[type=email], .swagger-ui textarea {
    background: #111827; color: #e5e7eb; border-color: #1e293b;
  }
  .swagger-ui .model-box, .swagger-ui section.models { background: #111827; border-color: #1e293b; }
  .swagger-ui section.models .model-container { background: #0b1220; }
  .swagger-ui table thead tr td, .swagger-ui table thead tr th { color: #7dd3fc; border-bottom-color: #1e293b; }
  .swagger-ui .response-col_status { color: #6ee7b7; }
  .swagger-ui .parameter__name { color: #e5e7eb; }
  .swagger-ui .parameter__type { color: #93c5fd; }
  .swagger-ui .markdown p, .swagger-ui .markdown li { color: #cbd5e1; }
</style>
"""


@app.get("/docs", include_in_schema=False)
def custom_swagger_ui_html() -> HTMLResponse:
    """Swagger UI con el mismo tema azul/verde que dashboard.html (ver
    _SWAGGER_THEME_CSS) en vez del blanco por defecto -- puramente visual, el
    endpoint sigue sirviendo la misma UI interactiva de siempre."""
    base_html = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Docs",
    ).body.decode()
    themed_html = base_html.replace("</head>", _SWAGGER_THEME_CSS + "</head>")
    return HTMLResponse(themed_html)


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
