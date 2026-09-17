"""
Tests de la integración de ticketing (case3_anomaly/service.py::_create_ticket_if_needed).

Cubre: sin TICKETING_WEBHOOK_URL configurada no se intenta nada (modo desarrollo/demo),
acciones que no ameritan intervención humana nunca disparan un ticket, con la URL configurada
se hace un POST real (mockeado acá), un webhook caído no rompe el triaje, y un webhook
reintentado con el mismo alert_id no crea un segundo ticket (idempotencia vía AlertTicket).
"""

import datetime as dt

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.cases.case3_anomaly import service
from app.cases.case3_anomaly.schemas import AlertWebhookPayload, TriageAction
from database import Base


@pytest.fixture()
def db_session():
    """Sesión de SQLAlchemy sobre una DB SQLite en memoria, aislada por test, con las tablas
    (incluida `alert_tickets`) creadas directamente vía create_all -- no depende del cliente
    FastAPI de tests/conftest.py porque estos tests llaman a `service._create_ticket_if_needed`
    directo, sin pasar por el endpoint."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def _payload(alert_id="alert-ticketing-1"):
    return AlertWebhookPayload(
        alert_id=alert_id,
        metric_name="checkout_conversion",
        z_score=4.2,
        triggered_at=dt.datetime(2026, 7, 9, 10, 0, 0),
    )


def test_sin_webhook_configurado_no_crea_ticket(monkeypatch, db_session):
    monkeypatch.setattr(service.settings, "TICKETING_WEBHOOK_URL", "")
    created = service._create_ticket_if_needed(1, TriageAction.escalate_engineering, _payload(), db_session)
    assert created is False


def test_accion_ignore_nunca_crea_ticket(monkeypatch, db_session):
    monkeypatch.setattr(service.settings, "TICKETING_WEBHOOK_URL", "https://ticketing.example.com/webhook")
    created = service._create_ticket_if_needed(1, TriageAction.ignore, _payload(), db_session)
    assert created is False


def test_con_webhook_configurado_hace_post_y_crea_ticket(monkeypatch, db_session):
    monkeypatch.setattr(service.settings, "TICKETING_WEBHOOK_URL", "https://ticketing.example.com/webhook")

    llamadas = []

    def _fake_post(url, json, timeout):
        llamadas.append((url, json, timeout))

        class _Resp:
            def raise_for_status(self):
                return None

        return _Resp()

    monkeypatch.setattr(service.httpx, "post", _fake_post)

    created = service._create_ticket_if_needed(7, TriageAction.escalate_business, _payload(), db_session)

    assert created is True
    assert len(llamadas) == 1
    url, body, _timeout = llamadas[0]
    assert url == "https://ticketing.example.com/webhook"
    assert body["company_id"] == 7
    assert body["alert_id"] == "alert-ticketing-1"
    assert body["action"] == "escalate_business"


def test_webhook_caido_no_rompe_y_devuelve_false(monkeypatch, db_session):
    monkeypatch.setattr(service.settings, "TICKETING_WEBHOOK_URL", "https://ticketing.example.com/webhook")

    def _fake_post_falla(url, json, timeout):
        raise httpx.ConnectError("no se pudo conectar", request=httpx.Request("POST", url))

    monkeypatch.setattr(service.httpx, "post", _fake_post_falla)

    created = service._create_ticket_if_needed(1, TriageAction.investigate_deep, _payload(), db_session)
    assert created is False


def test_webhook_reintentado_con_mismo_alert_id_no_duplica_ticket(monkeypatch, db_session):
    """Simula un reintento del webhook (mismo alert_id, típico ante timeouts de red de
    Datadog/Looker): la segunda llamada debe devolver True (ya existe ticket) sin volver a
    golpear la URL de ticketing."""
    monkeypatch.setattr(service.settings, "TICKETING_WEBHOOK_URL", "https://ticketing.example.com/webhook")

    llamadas = []

    def _fake_post(url, json, timeout):
        llamadas.append((url, json, timeout))

        class _Resp:
            def raise_for_status(self):
                return None

        return _Resp()

    monkeypatch.setattr(service.httpx, "post", _fake_post)

    payload = _payload(alert_id="alert-retry-1")

    primera = service._create_ticket_if_needed(3, TriageAction.escalate_engineering, payload, db_session)
    segunda = service._create_ticket_if_needed(3, TriageAction.escalate_engineering, payload, db_session)

    assert primera is True
    assert segunda is True
    assert len(llamadas) == 1  # el webhook externo sólo se llamó una vez


def test_webhook_caido_permite_reintento_posterior(monkeypatch, db_session):
    """Si el webhook falla, no se persiste ningún AlertTicket, así que un reintento posterior
    (mismo alert_id) sí debe volver a intentar la llamada -- no queda "atascado" en False."""
    monkeypatch.setattr(service.settings, "TICKETING_WEBHOOK_URL", "https://ticketing.example.com/webhook")

    estado = {"falla": True}
    llamadas = []

    def _fake_post(url, json, timeout):
        llamadas.append((url, json, timeout))
        if estado["falla"]:
            raise httpx.ConnectError("no se pudo conectar", request=httpx.Request("POST", url))

        class _Resp:
            def raise_for_status(self):
                return None

        return _Resp()

    monkeypatch.setattr(service.httpx, "post", _fake_post)

    payload = _payload(alert_id="alert-retry-2")

    primera = service._create_ticket_if_needed(4, TriageAction.escalate_engineering, payload, db_session)
    assert primera is False

    estado["falla"] = False
    segunda = service._create_ticket_if_needed(4, TriageAction.escalate_engineering, payload, db_session)
    assert segunda is True
    assert len(llamadas) == 2
