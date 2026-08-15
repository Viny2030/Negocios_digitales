def test_recommend_next_touchpoint(client, company_api_key):
    payload = {
        "prospect": {
            "prospect_id": "p-123",
            "channels_contacted": [],
            "messages_received": 0,
            "responses_count": 0,
            "days_since_last_contact": 5,
            "purchase_stage": "awareness",
            "unsubscribed": False,
        }
    }
    resp = client.post(
        "/api/v1/case1/recommend-next-touchpoint",
        json=payload,
        headers={"X-API-Key": company_api_key},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["prospect_id"] == "p-123"
    assert body["recommended_action"] in [
        "email_a",
        "email_b",
        "email_c",
        "retargeting_meta",
        "sms",
        "push",
        "call",
        "wait",
    ]


def test_requires_api_key(client):
    resp = client.post("/api/v1/case1/recommend-next-touchpoint", json={"prospect": {"prospect_id": "p-1"}})
    assert resp.status_code == 401


def test_weekly_email_cadence_blocks_second_email(client, company_api_key):
    """La heurística recomendaría email_a (no está en channels_contacted), pero se le
    mandó un email hace 2 días -> la restricción de 1 mail/semana debe bloquearlo."""
    payload = {
        "prospect": {
            "prospect_id": "p-cadence",
            "channels_contacted": ["email_b"],
            "messages_received": 1,
            "responses_count": 0,
            "days_since_last_contact": 2,
            "purchase_stage": "awareness",
            "unsubscribed": False,
            "days_since_last_email": 2,
        }
    }
    resp = client.post(
        "/api/v1/case1/recommend-next-touchpoint",
        json=payload,
        headers={"X-API-Key": company_api_key},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["recommended_action"] != "email_a"
    assert "cadencia_1_email_por_semana" in body["constraints_applied"]


def test_channel_budget_exhausted_falls_back_to_next_channel(client, company_api_key):
    """La heurística recomendaría retargeting_meta (email_a ya contactado, no es
    urgencia de decisión), pero el presupuesto semanal de ese canal ya se agotó."""
    payload = {
        "prospect": {
            "prospect_id": "p-budget",
            "channels_contacted": ["email_a"],
            "messages_received": 2,
            "responses_count": 0,
            "days_since_last_contact": 5,
            "purchase_stage": "awareness",
            "unsubscribed": False,
            "contacts_this_week": {"retargeting_meta": 5},
        },
        "channel_budget": {"retargeting_meta": 5},
    }
    resp = client.post(
        "/api/v1/case1/recommend-next-touchpoint",
        json=payload,
        headers={"X-API-Key": company_api_key},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["recommended_action"] != "retargeting_meta"
    assert "presupuesto_semanal_retargeting_meta_agotado" in body["constraints_applied"]
    assert "lp_seleccion_alternativa_factible" in body["constraints_applied"]
