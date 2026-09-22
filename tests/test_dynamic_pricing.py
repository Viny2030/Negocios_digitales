"""Tests de DynamicPricing API: ajuste de elasticidad, optimización con
restricciones y el endpoint end-to-end."""

from app.cases.dynamic_pricing.elasticity import fit_elasticity
from app.cases.dynamic_pricing.optimizer import optimize_price
from app.cases.dynamic_pricing.schemas import PricingObjective


def test_fit_elasticity_con_pocos_datos_usa_default():
    model = fit_elasticity([(10.0, 5), (10.0, 6)])  # < MIN_OBSERVATIONS_FOR_FIT
    assert model.is_default is True
    assert model.slope < 0


def test_fit_elasticity_recupera_pendiente_conocida():
    # Generamos datos sintéticos EXACTOS de quantity = 1000 * price**-2 (elasticidad -2)
    # a varios precios, para verificar que la regresión log-log recupera la pendiente.
    events = []
    for price in [5.0, 10.0, 20.0, 40.0, 80.0]:
        qty = round(1000 * price**-2)
        events.append((price, qty))

    model = fit_elasticity(events)
    assert model.is_default is False
    assert -2.2 < model.slope < -1.8


def test_fit_elasticity_pendiente_positiva_cae_a_default():
    # A más precio, más venta -- económicamente raro, no hay forma sensata de
    # optimizar sobre esto, así que debe caer al default.
    events = [(5.0, 1), (10.0, 5), (20.0, 20), (40.0, 80)]
    model = fit_elasticity(events)
    assert model.is_default is True


def test_optimize_price_respeta_margen_minimo():
    from app.cases.dynamic_pricing.elasticity import ElasticityModel

    elasticity = ElasticityModel(intercept=5.0, slope=-1.5, is_default=False)
    result = optimize_price(
        elasticity=elasticity,
        base_cost=100.0,
        current_price=101.0,  # casi sin margen -- la variación diaria por sí sola no alcanzaría
        min_margin_pct=0.15,
        max_daily_variation_pct=0.05,
        objective=PricingObjective.margin,
    )
    assert result.price >= 100.0 * 1.15 - 1e-6


def test_optimize_price_respeta_variacion_diaria_maxima():
    from app.cases.dynamic_pricing.elasticity import ElasticityModel

    elasticity = ElasticityModel(intercept=5.0, slope=-1.5, is_default=False)
    result = optimize_price(
        elasticity=elasticity,
        base_cost=10.0,  # margen mínimo bajo, no debería ser la restricción activa
        current_price=100.0,
        min_margin_pct=0.15,
        max_daily_variation_pct=0.05,
        objective=PricingObjective.margin,
    )
    assert 94.999 <= result.price <= 105.001


def test_optimize_price_objetivo_volumen_va_al_piso_del_rango():
    from app.cases.dynamic_pricing.elasticity import ElasticityModel

    elasticity = ElasticityModel(intercept=5.0, slope=-1.5, is_default=False)
    result = optimize_price(
        elasticity=elasticity,
        base_cost=10.0,
        current_price=100.0,
        min_margin_pct=0.15,
        max_daily_variation_pct=0.05,
        objective=PricingObjective.volume,
    )
    assert abs(result.price - 95.0) < 1e-6


def test_endpoint_price_devuelve_recomendacion_por_sku(client, company_api_key):
    payload = {
        "skus": [
            {
                "sku_id": "sku-1",
                "base_cost": 50.0,
                "current_price": 80.0,
                "stock": 100,
                "purchase_events": [
                    {"price": 70.0, "quantity_sold": 40},
                    {"price": 75.0, "quantity_sold": 30},
                    {"price": 80.0, "quantity_sold": 22},
                    {"price": 85.0, "quantity_sold": 15},
                    {"price": 90.0, "quantity_sold": 10},
                ],
                "min_margin_pct": 0.15,
                "max_daily_variation_pct": 0.05,
            }
        ],
        "objective": "margin",
    }
    resp = client.post("/api/v1/dynamic-pricing/price", json=payload, headers={"X-API-Key": company_api_key})
    assert resp.status_code == 200
    body = resp.json()
    result = body["results"][0]
    assert result["sku_id"] == "sku-1"
    assert result["used_default_elasticity"] is False
    assert 76.0 <= result["recommended_price"] <= 84.0  # dentro de la banda de variación diaria
    assert result["expected_margin_pct"] > 0.15 - 1e-6


def test_endpoint_price_requiere_api_key(client):
    payload = {"skus": [{"sku_id": "a", "base_cost": 10, "current_price": 20, "stock": 5}]}
    resp = client.post("/api/v1/dynamic-pricing/price", json=payload)
    assert resp.status_code == 401


def test_endpoint_price_rechaza_sku_ids_duplicados(client, company_api_key):
    payload = {
        "skus": [
            {"sku_id": "dup", "base_cost": 10, "current_price": 20, "stock": 5},
            {"sku_id": "dup", "base_cost": 12, "current_price": 22, "stock": 3},
        ]
    }
    resp = client.post("/api/v1/dynamic-pricing/price", json=payload, headers={"X-API-Key": company_api_key})
    assert resp.status_code == 422
