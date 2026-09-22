"""Tests del punto de entrada: /health y el /docs con tema custom (ver app/main.py)."""


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_docs_usa_tema_custom(client):
    resp = client.get("/docs")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    # No probamos colores exactos (frágil ante retoques de paleta) -- solo que el
    # bloque de estilo custom de app/main.py efectivamente se está inyectando en vez
    # de servir el Swagger UI default sin tema.
    assert "swagger-ui .opblock.opblock-get" in resp.text


def test_openapi_json_disponible(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert resp.json()["info"]["title"]
