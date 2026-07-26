def test_allocate_respects_total_mentorship_budget(client, company_api_key):
    payload = {
        "startups": [
            {"startup_id": "s1", "traction_score": 0.9, "team_score": 0.8, "market_score": 0.7, "runway_months": 6},
            {"startup_id": "s2", "traction_score": 0.3, "team_score": 0.4, "market_score": 0.5, "runway_months": 12},
            {"startup_id": "s3", "traction_score": 0.6, "team_score": 0.6, "market_score": 0.6, "runway_months": 9},
        ],
        "resources": {
            "mentorship_hours_available": 30,
            "followon_capital_available": 0,
            "strategic_connections_available": 1,
        },
    }
    resp = client.post("/api/v1/case4/allocate", json=payload, headers={"X-API-Key": company_api_key})
    assert resp.status_code == 200
    body = resp.json()

    total_mentorship = sum(a["mentorship_hours"] for a in body["allocations"])
    assert total_mentorship <= 30 + 1e-6

    best = max(body["allocations"], key=lambda a: a["expected_value_score"])
    assert best["startup_id"] == "s1"
    assert best["mentorship_hours"] == max(a["mentorship_hours"] for a in body["allocations"])
    assert best["mentorship_hours"] > 0


def test_allocate_respects_concentration_cap(client, company_api_key):
    payload = {
        "startups": [
            {"startup_id": "s1", "traction_score": 0.9, "team_score": 0.9, "market_score": 0.9, "runway_months": 6},
            {"startup_id": "s2", "traction_score": 0.1, "team_score": 0.1, "market_score": 0.1, "runway_months": 6},
        ],
        "resources": {
            "mentorship_hours_available": 100,
            "followon_capital_available": 0,
            "strategic_connections_available": 0,
            "max_mentorship_share_per_startup": 0.5,
        },
    }
    resp = client.post("/api/v1/case4/allocate", json=payload, headers={"X-API-Key": company_api_key})
    assert resp.status_code == 200
    body = resp.json()

    for allocation in body["allocations"]:
        assert allocation["mentorship_hours"] <= 50 + 1e-6
    assert "tope_concentracion_mentoria" in body["constraints_applied"]


def test_allocate_limits_strategic_connections(client, company_api_key):
    payload = {
        "startups": [
            {"startup_id": f"s{i}", "traction_score": 0.5, "team_score": 0.5, "market_score": 0.5, "runway_months": 6}
            for i in range(5)
        ],
        "resources": {
            "mentorship_hours_available": 10,
            "followon_capital_available": 0,
            "strategic_connections_available": 2,
        },
    }
    resp = client.post("/api/v1/case4/allocate", json=payload, headers={"X-API-Key": company_api_key})
    assert resp.status_code == 200
    body = resp.json()

    activated = [a for a in body["allocations"] if a["strategic_connection_activated"]]
    assert len(activated) <= 2


def test_allocate_requires_api_key(client):
    payload = {
        "startups": [{"startup_id": "s1", "traction_score": 0.5, "team_score": 0.5, "market_score": 0.5, "runway_months": 6}],
        "resources": {"mentorship_hours_available": 10},
    }
    resp = client.post("/api/v1/case4/allocate", json=payload)
    assert resp.status_code == 401