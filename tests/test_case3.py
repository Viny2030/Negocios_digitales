def test_alert_webhook_returns_decision(client, company_api_key):
    payload = {
        "alert_id": "alert-1",
        "metric_name": "checkout_conversion",
        "z_score": 3.5,
        "pct_change": -12.5,
        "triggered_at": "2026-07-09T10:00:00",
        "business_context": "checkout",
        "historical_similar_count": 1,
        "estimated_financial_impact": 8000,
    }
    resp = client.post("/api/v1/case3/alert-webhook", json=payload, headers={"X-API-Key": company_api_key})
    assert resp.status_code == 200
    body = resp.json()
    assert body["alert_id"] == "alert-1"
    assert body["action"] in [
        "ignore", "investigate_light", "investigate_deep",
        "escalate_engineering", "escalate_business", "merge_related",
    ]
