"""
Tests de la integración de ticketing (case3_anomaly/service.py::_create_ticket_if_needed).

Cubre: sin TICKETING_WEBHOOK_URL configurada no se intenta nada (modo desarrollo/demo),
acciones que no ameritan intervención humana nunca disparan un ticket, con la URL configurada
se hace un POST real (mockeado acá) y un webhook caído no rompe el triaje.
"""

import datetime as dt

import httpx

from app.cases.case3_anomaly import service
from app.cases.case3_anomaly.schemas import AlertWebhookPayload, TriageAction


def _payload():
    return AlertWebhookPayload(
        alert_id="alert-ticketing-1",
        metric_name="checkout_conversion",
        z_score=4.2,
        triggered_at=dt.datetime(2026, 7, 9, 10, 0, 0),
    )


def test_sin_webhook_configurado_no_crea_ticket(monkeypatch):
    monkeypatch.setattr(service.settings, "TICKETING_WEBHOOK_URL", "")
    created = service._create_ticket_if_needed(1, TriageAction.escalate_engineering, _payload())
    assert created is False


def test_accion_ignore_nunca_crea_ticket(monkeypatch):
    monkeypatch.setattr(service.settings, "TICKETING_WEBHOOK_URL", "https://ticketing.example.com/webhook")
    created = service._create_ticket_if_needed(1, TriageAction.ignore, _payload())
    assert created is False


def test_con_webhook_configurado_hace_post_y_crea_ticket(monkeypatch):
    monkeypatch.setattr(service.settings, "TICKETING_WEBHOOK_URL", "https://ticketing.example.com/webhook")

    llamadas = []

    def _fake_post(url, json, timeout):
        llamadas.append((url, json, timeout))

        class _Resp:
            def raise_for_status(self):
                return None

        return _Resp()

    monkeypatch.setattr(service.httpx, "post", _fake_post)

    created = service._create_ticket_if_needed(7, TriageAction.escalate_business, _payload())

    assert created is True
    assert len(llamadas) == 1
    url, body, _timeout = llamadas[0]
    assert url == "https://ticketing.example.com/webhook"
    assert body["company_id"] == 7
    assert body["alert_id"] == "alert-ticketing-1"
    assert body["action"] == "escalate_business"


def test_webhook_caido_no_rompe_y_devuelve_false(monkeypatch):
    monkeypatch.setattr(service.settings, "TICKETING_WEBHOOK_URL", "https://ticketing.example.com/webhook")

    def _fake_post_falla(url, json, timeout):
        raise httpx.ConnectError("no se pudo conectar", request=httpx.Request("POST", url))

    monkeypatch.setattr(service.httpx, "post", _fake_post_falla)

    created = service._create_ticket_if_needed(1, TriageAction.investigate_deep, _payload())
    assert created is False
