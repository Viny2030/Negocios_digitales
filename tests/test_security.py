"""Tests de los controles de seguridad: API keys con hash, límites de pedidos, topes de
tamaño y nombres de archivo."""

import hashlib
import os
import uuid

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.cases.lead_match.matching import match_leads
from app.cases.lead_match.schemas import LeadFeatures, MatchRequest, SalesRep
from app.cases.smart_route.schemas import RoutingRequest
from app.main import app
from app.middleware import BodySizeLimitMiddleware
from app.models.company import Company
from app.services import dataset_ingestion
from app.services.dataset_ingestion import safe_filename
from database import get_db
from rate_limit import GlobalRateLimitMiddleware, client_ip, limiter, rate_limit_key
from rate_limit import settings as rate_limit_settings

CSV_MINIMO = b"lead_id,origin\n1,a\n"


@pytest.fixture(autouse=True)
def _reset_limiter():
    limiter.reset()
    yield
    limiter.reset()


def _register(client, **kwargs):
    email = f"{uuid.uuid4().hex[:10]}@empresa.com"
    return client.post("/api/v1/companies", json={"name": "Empresa", "contact_email": email}, **kwargs)


def test_la_base_guarda_solo_el_hash_de_la_key(client):
    body = _register(client).json()
    db = next(app.dependency_overrides[get_db]())
    company = db.get(Company, body["id"])

    assert company.api_key_hash == hashlib.sha256(body["api_key"].encode()).hexdigest()
    assert company.api_key_prefix == body["api_key"][:11]
    assert not hasattr(company, "api_key")


def test_me_no_devuelve_la_key(client):
    body = _register(client).json()
    me = client.get("/api/v1/companies/me", headers={"X-API-Key": body["api_key"]})

    assert me.status_code == 200
    assert "api_key" not in me.json()
    assert me.json()["api_key_prefix"] == body["api_key_prefix"]


def test_rotar_invalida_la_key_anterior(client):
    old_key = _register(client).json()["api_key"]

    rotated = client.post("/api/v1/companies/me/rotate-key", headers={"X-API-Key": old_key})
    assert rotated.status_code == 200
    new_key = rotated.json()["api_key"]

    assert new_key != old_key
    assert client.get("/api/v1/companies/me", headers={"X-API-Key": old_key}).status_code == 401
    assert client.get("/api/v1/companies/me", headers={"X-API-Key": new_key}).status_code == 200


def test_rotar_requiere_key_valida(client):
    assert client.post("/api/v1/companies/me/rotate-key", headers={"X-API-Key": "pk_falsa"}).status_code == 401


def test_nombre_de_empresa_demasiado_largo_da_422(client):
    resp = client.post("/api/v1/companies", json={"name": "x" * 256, "contact_email": "a@empresa.com"})
    assert resp.status_code == 422


def test_el_registro_se_limita_aunque_se_invente_una_key_distinta(client, monkeypatch):
    monkeypatch.setattr(rate_limit_settings, "REGISTRATION_RATE_LIMIT", "2/minute")

    codes = [_register(client, headers={"X-API-Key": uuid.uuid4().hex}).status_code for _ in range(4)]

    assert codes == [201, 201, 429, 429]


def test_techo_por_ip_frena_keys_inventadas():
    mini = FastAPI()
    mini.add_middleware(GlobalRateLimitMiddleware, ip_limit="3/minute")

    @mini.get("/ping")
    def ping():
        return {"ok": True}

    with TestClient(mini) as c:
        codes = [c.get("/ping", headers={"X-API-Key": uuid.uuid4().hex}).status_code for _ in range(5)]

    assert codes == [200, 200, 200, 429, 429]


def test_el_limite_por_defecto_se_aplica_en_la_app_real(client, company_api_key):
    """Contra la app real a propósito: con FastAPI 0.162, SlowAPIMiddleware dejó de ver las
    rutas de include_router y este límite se apagó sin que fallara ningún test."""
    headers = {"X-API-Key": company_api_key}
    codes = [client.get("/api/v1/companies/me", headers=headers).status_code for _ in range(125)]

    assert 429 in codes
    assert codes.index(429) <= 120
    assert client.get("/api/v1/companies/me", headers={"X-API-Key": "pk_otra"}).status_code == 401


def test_los_limites_de_un_endpoint_se_aplican_en_la_app_real(client, company_api_key):
    headers = {"X-API-Key": company_api_key}
    codes = [
        client.post("/api/v1/lead-match/train", headers=headers, json={"dataset_id": 999}).status_code
        for _ in range(7)
    ]
    assert codes == [404] * 5 + [429] * 2


def test_upload_sin_content_length_que_supera_el_tope_da_413(client, company_api_key):
    def chunks():
        yield b'--x\r\nContent-Disposition: form-data; name="file"; filename="b.csv"\r\n\r\nlead_id,origin\n'
        for _ in range(30):
            yield b"1,a\n" * 256 * 1024
        yield b"\r\n--x--\r\n"

    resp = client.post(
        "/api/v1/datasets/lead_match",
        headers={"X-API-Key": company_api_key, "Content-Type": "multipart/form-data; boundary=x"},
        content=chunks(),
    )
    assert resp.status_code == 413


def _request_with_forwarded(value: str) -> Request:
    headers = [(b"x-forwarded-for", value.encode())]
    return Request({"type": "http", "headers": headers, "client": ("10.0.0.1", 1234)})


def test_sin_proxies_de_confianza_se_ignora_x_forwarded_for(monkeypatch):
    monkeypatch.setattr(rate_limit_settings, "TRUSTED_PROXY_HOPS", 0)
    assert client_ip(_request_with_forwarded("1.1.1.1")) == "10.0.0.1"


def test_con_un_proxy_no_se_puede_falsificar_la_ip(monkeypatch):
    monkeypatch.setattr(rate_limit_settings, "TRUSTED_PROXY_HOPS", 1)
    assert client_ip(_request_with_forwarded("6.6.6.6, 2.2.2.2")) == "2.2.2.2"


def test_la_clave_del_limitador_no_contiene_la_key():
    request = Request({"type": "http", "headers": [(b"x-api-key", b"pk_secreta")], "client": ("10.0.0.1", 1)})
    assert "pk_secreta" not in rate_limit_key(request)


@pytest.fixture()
def body_limited_client():
    mini = FastAPI()
    mini.add_middleware(BodySizeLimitMiddleware, max_json_bytes=100, max_upload_bytes=100)

    @mini.post("/eco")
    async def eco(request: Request):
        return {"size": len(await request.body())}

    with TestClient(mini) as c:
        yield c


def test_cuerpo_declarado_demasiado_grande_da_413(body_limited_client):
    assert body_limited_client.post("/eco", content=b"x" * 101).status_code == 413
    assert body_limited_client.post("/eco", content=b"x" * 100).status_code == 200


def test_cuerpo_sin_content_length_tambien_se_corta(body_limited_client):
    def chunks():
        yield b"x" * 60
        yield b"x" * 60

    assert body_limited_client.post("/eco", content=chunks()).status_code == 413


def test_upload_que_supera_el_tope_da_413_y_no_queda_guardado(client, company_api_key, monkeypatch):
    monkeypatch.setattr(dataset_ingestion.settings, "MAX_UPLOAD_BYTES", 10)
    headers = {"X-API-Key": company_api_key}

    resp = client.post("/api/v1/datasets/lead_match", headers=headers, files={"file": ("a.csv", CSV_MINIMO)})

    assert resp.status_code == 413
    assert client.get("/api/v1/datasets/lead_match", headers=headers).json() == []


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("datos.csv", "datos.csv"),
        ("../../../evil.csv", "evil.csv"),
        ("..\\..\\evil.csv", "evil.csv"),
        ("C:\\temp\\evil.csv", "evil.csv"),
        ("..", "dataset"),
        ("a:b?.csv", "a_b_.csv"),
        ("ventas año.csv", "ventas año.csv"),
    ],
)
def test_safe_filename(original, expected):
    assert safe_filename(original) == expected


def test_upload_con_ruta_en_el_nombre_queda_en_la_carpeta_de_la_empresa(client, company_api_key):
    resp = client.post(
        "/api/v1/datasets/lead_match",
        headers={"X-API-Key": company_api_key},
        files={"file": ("../../../evil.csv", CSV_MINIMO)},
    )
    assert resp.status_code == 201
    assert resp.json()["name"] == "evil.csv"

    db = next(app.dependency_overrides[get_db]())
    from app.models.dataset import Dataset

    stored = os.path.normpath(db.get(Dataset, resp.json()["id"]).file_path)
    expected_dir = os.path.normpath(
        os.path.join(dataset_ingestion.settings.DATASETS_DIR, str(resp.json()["company_id"]), "lead_match")
    )
    assert os.path.dirname(stored) == expected_dir


def _lead(i: int) -> dict:
    return {"lead_id": f"l{i}", "origin": "a", "time_on_site_minutes": 1, "company_size": 1}


def test_demasiadas_paradas_da_error():
    stop = {"location": {"lat": 0, "lon": 0}, "demand": 1, "time_window_start_min": 0, "time_window_end_min": 9}
    stops = [{**stop, "stop_id": f"s{i}"} for i in range(101)]
    with pytest.raises(ValidationError):
        RoutingRequest(depot={"lat": 0, "lon": 0}, stops=stops, vehicles=[{"vehicle_id": "v", "capacity": 1}])


def test_capacidad_enorme_no_agranda_el_pedido():
    MatchRequest(leads=[_lead(0)], reps=[{"rep_id": "r", "capacity": 10**9}])


def test_demasiadas_combinaciones_da_error():
    reps = [{"rep_id": f"r{i}", "capacity": 1000} for i in range(3)]
    with pytest.raises(ValidationError, match="demasiado grande"):
        MatchRequest(leads=[_lead(i) for i in range(1000)], reps=reps)


def test_recortar_cupos_no_cambia_la_asignacion():
    leads = [LeadFeatures(**_lead(i)) for i in range(3)]
    probabilities = {lead.lead_id: 0.5 for lead in leads}

    assignments, unmatched = match_leads(
        leads, probabilities, [SalesRep(rep_id="r", capacity=10**9)], specialty_bonus=1.2
    )

    assert {a.lead_id for a in assignments} == {"l0", "l1", "l2"}
    assert unmatched == []
