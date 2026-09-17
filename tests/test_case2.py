import pytest


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


def test_get_price_rejects_min_margin_pct_fuera_de_rango(client, company_api_key):
    """Antes de la validación de rango, un min_margin_pct muy negativo (ej. -5.0) podía
    producir un piso de precio negativo y, en cascada, un final_price negativo. Ahora
    pydantic lo rechaza con 422 antes de llegar a calcular ningún precio."""
    payload = {
        "product_id": "sku-1",
        "base_cost": 100.0,
        "user_segment": "premium",
        "inventory_level_pct": 20.0,
        "min_margin_pct": -5.0,
    }
    resp = client.post("/api/v1/case2/get-price", json=payload, headers={"X-API-Key": company_api_key})
    assert resp.status_code == 422


def test_get_price_rejects_max_multiplier_fuera_de_rango(client, company_api_key):
    payload = {
        "product_id": "sku-1",
        "base_cost": 100.0,
        "user_segment": "premium",
        "inventory_level_pct": 20.0,
        "max_multiplier": 10_000.0,
    }
    resp = client.post("/api/v1/case2/get-price", json=payload, headers={"X-API-Key": company_api_key})
    assert resp.status_code == 422


def test_get_price_rejects_max_multiplier_menor_o_igual_a_min_multiplier(client, company_api_key):
    payload = {
        "product_id": "sku-1",
        "base_cost": 100.0,
        "user_segment": "premium",
        "inventory_level_pct": 20.0,
        "max_multiplier": 1.0,
        "min_multiplier": 1.0,
    }
    resp = client.post("/api/v1/case2/get-price", json=payload, headers={"X-API-Key": company_api_key})
    assert resp.status_code == 422


def test_get_price_competitor_avg_price_cero_se_trata_como_valor_real(client, company_api_key):
    """`competitor_avg_price=0.0` es un valor explícito (ej. "no hay referencia de
    mercado, asumir 0"), no "no vino el campo" -- antes `if request.competitor_avg_price:`
    lo trataba como falsy y se saltaba el chequeo de coherencia competitiva por completo."""
    payload = {
        "product_id": "sku-1",
        "base_cost": 100.0,
        "user_segment": "premium",
        "inventory_level_pct": 20.0,
        "competitor_avg_price": 0.0,
        "min_margin_pct": 0.10,
    }
    resp = client.post("/api/v1/case2/get-price", json=payload, headers={"X-API-Key": company_api_key})
    assert resp.status_code == 200
    body = resp.json()
    assert "coherencia_competitiva" in body["constraints_applied"]
    assert body["final_price"] >= 0.0


def test_get_price_final_price_nunca_negativo(client, company_api_key):
    payload = {
        "product_id": "sku-1",
        "base_cost": 100.0,
        "user_segment": "premium",
        "inventory_level_pct": 20.0,
        "competitor_avg_price": 0.0,
        "min_margin_pct": 0.10,
    }
    resp = client.post("/api/v1/case2/get-price", json=payload, headers={"X-API-Key": company_api_key})
    assert resp.status_code == 200
    assert resp.json()["final_price"] >= 0.0


def test_train_endpoint_rechaza_algorithm_dqn(client, company_api_key):
    """TrainRequest.algorithm ahora es Literal["ppo"]: el espacio de acciones de
    PricingEnv es continuo (Box) y SB3's DQN no lo soporta. "dqn" debe rechazarse en
    la validación del esquema, en vez de llegar a instanciar DQN y crashear."""
    payload = {"dataset_id": 1, "algorithm": "dqn"}
    resp = client.post("/api/v1/case2/train", json=payload, headers={"X-API-Key": company_api_key})
    assert resp.status_code == 422


def test_train_for_company_rechaza_algorithm_dqn_con_error_claro():
    """Defensa en profundidad a nivel del script de entrenamiento (no solo del esquema
    HTTP): un llamado directo a train_for_company con algorithm="dqn" debe fallar con un
    ValueError explicativo, antes de tocar la base de datos o instanciar el modelo SB3,
    en vez de un crash oscuro dentro de stable-baselines3."""
    from scripts.train_case2 import train_for_company

    with pytest.raises(ValueError, match="(?i)dqn|continu"):
        train_for_company(company_id=999_999, dataset_id=999_999, algorithm="dqn")
