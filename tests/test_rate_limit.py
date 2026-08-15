"""
Test del mecanismo de rate limiting (ver rate_limit.py): confirma que el límite se
aplica por empresa (clave = header X-API-Key) y no globalmente ni por IP, y que superar
el límite devuelve 429.

Se prueba contra una mini app FastAPI standalone en vez del app real (app.main) porque
los endpoints reales que tienen un @limiter.limit explícito son los de /train, que
necesitan Celery + Redis corriendo -- acá se valida el mecanismo en sí, aislado.
"""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from rate_limit import limiter


def _build_test_app():
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    @app.get("/ping")
    @limiter.limit("3/minute")
    def ping(request: Request):
        return {"ok": True}

    return app


# El `limiter` es un singleton a nivel de módulo (el mismo que usa app/main.py en producción).
# slowapi registra cada `@limiter.limit(...)` bajo una clave basada en el nombre de la función
# decorada, y ACUMULA registros en vez de reemplazarlos -- si `_build_test_app()` se llamara una
# vez por test, la segunda llamada duplicaría el límite "3/minute" para la ruta "ping" contra el
# mismo limiter, y cada request contaría doble (justo lo que causaba que el segundo test fallara
# con 429 en la primera request). Por eso la app se construye UNA sola vez para todo el módulo,
# igual que en producción, y entre tests solo se resetea el storage (los contadores), no el
# registro de límites.
_client = TestClient(_build_test_app())


@pytest.fixture(autouse=True)
def _reset_limiter_storage():
    """Limpia los contadores del `limiter` compartido antes y después de cada test, para que
    el orden de ejecución de pytest no afecte el resultado."""
    limiter.reset()
    yield
    limiter.reset()


def test_rate_limit_bloquea_al_superar_el_limite():
    codes = [_client.get("/ping", headers={"X-API-Key": "pk_test_empresa_rl_1"}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]


def test_rate_limit_aisla_por_empresa():
    """Dos empresas (dos API keys distintas) no deben compartir el contador."""
    # Agota el límite de la empresa A
    for _ in range(3):
        assert _client.get("/ping", headers={"X-API-Key": "pk_test_empresa_rl_A"}).status_code == 200
    assert _client.get("/ping", headers={"X-API-Key": "pk_test_empresa_rl_A"}).status_code == 429

    # La empresa B, en la misma ruta, arranca su propio contador en cero
    assert _client.get("/ping", headers={"X-API-Key": "pk_test_empresa_rl_B"}).status_code == 200
