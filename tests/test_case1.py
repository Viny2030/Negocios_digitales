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
        "email_a", "email_b", "email_c", "retargeting_meta", "sms", "push", "call", "wait",
    ]


def test_requires_api_key(client):
    resp = client.post("/api/v1/case1/recommend-next-touchpoint", json={"prospect": {"prospect_id": "p-1"}})
    assert resp.status_code == 401
