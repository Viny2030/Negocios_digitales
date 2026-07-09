def test_get_price_respects_min_margin(client, company_api_key):
    payload = {
        "product_id": "sku-1",
        "base_cost": 100.0,
        "user_segment": "premium",
        "inventory_level_pct": 20.0,
        "competitor_avg_price": 150.0,
        "demand_index": 0.8,
        "min_margin_pct": 0.15,
    }
    resp = client.post("/api/v1/case2/get-price", json=payload, headers={"X-API-Key": company_api_key})
    assert resp.status_code == 200
    body = resp.json()
    assert body["final_price"] >= 100.0 * 1.15 - 1e-6
    assert body["latency_ms"] >= 0
