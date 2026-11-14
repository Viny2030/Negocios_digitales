"""Un test por cada defecto de producto que se reprodujo en la revisión del 09/11/2026."""

import io
import itertools
import random

import pytest

from app.cases.dynamic_pricing.elasticity import ElasticityModel
from app.cases.dynamic_pricing.optimizer import optimize_price
from app.cases.dynamic_pricing.schemas import PricingObjective
from app.cases.lead_match.schemas import LeadFeatures
from app.cases.lead_match.scoring import ConversionScorer
from app.cases.lead_match.train import _decide_promotion
from app.cases.smart_route.schemas import RoutingRequest
from app.cases.smart_route.solver import solve

# ---------------------------------------- SmartRoute


def _route_request(stops, vehicles) -> RoutingRequest:
    return RoutingRequest(
        depot={"lat": 0, "lon": 0},
        stops=[
            {
                "stop_id": stop_id,
                "location": {"lat": lat, "lon": 0},
                "demand": demand,
                "time_window_start_min": start,
                "time_window_end_min": end,
            }
            for stop_id, lat, demand, start, end in stops
        ],
        vehicles=vehicles,
    )


def _assigned(response) -> set[str]:
    return {stop.stop_id for route in response.routes for stop in route.stops}


def test_parada_descartada_usa_un_vehiculo_libre():
    request = _route_request(
        [("n", 0.05, 1, 0, 20), ("s", -0.05, 1, 0, 20)],
        [{"vehicle_id": "v1", "capacity": 10}, {"vehicle_id": "v2", "capacity": 10}],
    )
    response = solve(request)
    assert _assigned(response) == {"n", "s"}
    assert response.unassigned_stops == []
    assert len(response.routes) == 2


def test_flota_heterogenea_usa_toda_su_capacidad():
    request = _route_request(
        [(f"s{i}", 0.001 * i, 10, 0, 480) for i in range(1, 7)],
        [{"vehicle_id": name, "capacity": cap} for name, cap in [("big", 30), ("a", 10), ("b", 10), ("c", 10)]],
    )
    response = solve(request)
    assert response.unassigned_stops == []
    assert all(route.load_used <= route.capacity for route in response.routes)


@pytest.mark.parametrize(("near", "far"), [("cerca", "lejos"), ("z_cerca", "a_lejos")])
def test_el_resultado_no_depende_del_nombre_de_las_paradas(near, far):
    request = _route_request(
        [(near, 0.01, 1, 60, 200), (far, 0.20, 1, 0, 50)],
        [{"vehicle_id": "v1", "capacity": 10}],
    )
    response = solve(request)
    assert response.unassigned_stops == []
    assert [stop.stop_id for stop in response.routes[0].stops] == [far, near]


def test_el_resultado_no_depende_del_orden_de_las_paradas():
    stops = [("a", 0.05, 1, 0, 20), ("b", -0.05, 1, 0, 20), ("c", 0.01, 1, 0, 480)]
    vehicles = [{"vehicle_id": "v1", "capacity": 10}, {"vehicle_id": "v2", "capacity": 10}]
    for permutation in itertools.permutations(stops):
        response = solve(_route_request(list(permutation), vehicles))
        assert _assigned(response) == {"a", "b", "c"}


def test_la_duracion_incluye_el_regreso_y_respeta_el_tope():
    # A ~55,6 km y 30 km/h: ~111 min de ida, 5 de servicio y ~111 de regreso.
    stops = [("s1", 0.5, 1, 0, 480)]
    vehicle = {"vehicle_id": "v1", "capacity": 10, "average_speed_kmh": 30}

    with_room = solve(_route_request(stops, [{**vehicle, "max_route_duration_min": 300}]))
    assert 225 < with_room.routes[0].total_duration_min < 230

    too_short = solve(_route_request(stops, [{**vehicle, "max_route_duration_min": 150}]))
    assert too_short.routes == []
    assert too_short.unassigned_stops == ["s1"]


def test_elige_el_vehiculo_mas_chico_que_alcanza():
    response = solve(
        _route_request(
            [("s1", 0.01, 5, 0, 240)],
            [{"vehicle_id": "grande", "capacity": 100}, {"vehicle_id": "chico", "capacity": 10}],
        )
    )
    assert response.routes[0].vehicle_id == "chico"


def test_ninguna_ruta_rompe_las_reglas_en_un_caso_grande():
    rng = random.Random(3)
    stops = []
    for i in range(100):
        start = rng.randint(0, 300)
        stops.append((f"s{i}", rng.uniform(-0.1, 0.1), rng.randint(1, 5), start, start + rng.randint(20, 120)))
    request = _route_request(stops, [{"vehicle_id": f"v{i}", "capacity": rng.randint(10, 60)} for i in range(10)])
    by_id = {stop.stop_id: stop for stop in request.stops}

    response = solve(request)

    assert _assigned(response) | set(response.unassigned_stops) == set(by_id)
    assert not _assigned(response) & set(response.unassigned_stops)
    for route in response.routes:
        assert route.load_used <= route.capacity
        assert route.total_duration_min <= 480
        for stop in route.stops:
            assert stop.arrival_min <= by_id[stop.stop_id].time_window_end_min + 0.01


# ---------------------------------------- DynamicPricing

# quantity = exp(a) * price**b: a este precio de 150 se venden unas 22 unidades.
DEMAND = ElasticityModel(intercept=31.0, slope=-5.57, is_default=False)
NO_DATA = ElasticityModel(intercept=3.0, slope=-1.2, is_default=True)


def _optimize(elasticity=DEMAND, objective=PricingObjective.margin, **kwargs):
    params = {
        "base_cost": 100.0,
        "current_price": 150.0,
        "min_margin_pct": 0.15,
        "max_daily_variation_pct": 0.10,
    }
    return optimize_price(elasticity=elasticity, objective=objective, **{**params, **kwargs})


def test_sin_datos_no_sube_el_precio():
    result = _optimize(elasticity=NO_DATA, stock=10)
    assert result.price == 150.0
    assert result.expected_quantity is None
    assert any(c.startswith("sin_datos") for c in result.constraints_applied)


def test_sin_datos_igual_respeta_el_margen_minimo():
    assert _optimize(elasticity=NO_DATA, current_price=110.0).price == pytest.approx(115.0)


def test_sin_stock_no_espera_ventas():
    result = _optimize(stock=0)
    assert result.price == 150.0
    assert result.expected_quantity == 0.0
    assert any(c.startswith("sin_stock") for c in result.constraints_applied)


def test_la_cantidad_esperada_no_supera_el_stock():
    assert _optimize(stock=5).expected_quantity <= 5


def test_con_poco_stock_el_precio_sube():
    assert _optimize(stock=5).price > _optimize(stock=100000).price


def test_objetivo_volumen_no_regala_stock():
    plenty = _optimize(objective=PricingObjective.volume, stock=100000)
    scarce = _optimize(objective=PricingObjective.volume, stock=20)
    assert plenty.price == pytest.approx(135.0)
    assert scarce.price > plenty.price
    assert scarce.expected_quantity == pytest.approx(20)


def test_el_competidor_pone_un_techo():
    result = _optimize(elasticity=ElasticityModel(10.0, -0.5, False), competitor_price=140.0)
    assert result.price == pytest.approx(154.0, abs=0.01)  # 140 * 1.10, por debajo del techo de 165


def test_el_margen_minimo_le_gana_al_competidor():
    result = _optimize(competitor_price=50.0)
    assert result.price == pytest.approx(135.0)
    assert any("competidor" in c and c.startswith("conflicto") for c in result.constraints_applied)


def test_endpoint_informa_el_margen_con_las_dos_bases(client, company_api_key):
    sku = {
        "sku_id": "a",
        "base_cost": 100,
        "current_price": 115,
        "stock": 10,
        "max_daily_variation_pct": 0.001,
        "min_margin_pct": 0.15,
    }
    resp = client.post(
        "/api/v1/dynamic-pricing/price", json={"skus": [sku]}, headers={"X-API-Key": company_api_key}
    )
    result = resp.json()["results"][0]
    assert result["recommended_price"] == 115
    assert result["expected_margin_on_cost_pct"] == pytest.approx(0.15)
    assert result["expected_margin_pct"] == pytest.approx(0.1304)
    assert result["expected_quantity"] is None
    assert result["expected_profit"] is None


# ---------------------------------------- LeadMatch

HEADER = "origin,time_on_site_minutes,company_size,converted\n"


def _signal_csv(n=200) -> str:
    """Compra quien pasa más de 15 minutos en el sitio."""
    rng = random.Random(1)
    rows = [(rng.uniform(0, 30), rng.randint(1, 500)) for _ in range(n)]
    return HEADER + "".join(f"ads,{t:.2f},{s},{int(t > 15)}\n" for t, s in rows)


def _noise_csv(n=200) -> str:
    rng = random.Random(2)
    return HEADER + "".join(
        f"ads,{rng.uniform(0, 30):.2f},{rng.randint(1, 500)},{rng.randint(0, 1)}\n" for _ in range(n)
    )


def _inverted_csv(n=200) -> str:
    """Aprende lo contrario de _signal_csv: sobre esos datos sería un modelo pésimo."""
    return _signal_csv(n).replace(",1\n", ",X\n").replace(",0\n", ",1\n").replace(",X\n", ",0\n")


def _train(client, headers, csv_text):
    upload = client.post(
        "/api/v1/datasets/lead_match",
        headers=headers,
        files={"file": ("leads.csv", io.BytesIO(csv_text.encode()), "text/csv")},
    )
    assert upload.status_code == 201, upload.text
    return client.post("/api/v1/lead-match/train", headers=headers, json={"dataset_id": upload.json()["id"]})


@pytest.fixture()
def headers(company_api_key):
    return {"X-API-Key": company_api_key}


def test_un_modelo_de_ruido_no_se_promueve(client, headers):
    body = _train(client, headers, _noise_csv()).json()
    assert body["promoted"] is False
    assert body["current_version_id"] is None
    assert "azar" in body["promotion_reason"]


def test_un_modelo_de_ruido_no_reemplaza_al_vigente(client, headers):
    good = _train(client, headers, _signal_csv()).json()
    assert good["promoted"] is True

    noise = _train(client, headers, _noise_csv()).json()

    assert noise["promoted"] is False
    assert noise["current_version_id"] == good["version_id"]


@pytest.mark.parametrize(
    ("new_auc", "current_auc", "expected"),
    [
        (0.70, 0.80, False),  # útil, pero peor que el vigente
        (0.795, 0.80, True),  # dentro del margen de ruido
        (0.85, 0.80, True),
        (0.70, None, True),  # primer modelo
        (0.54, None, False),  # casi azar
        (None, 0.80, False),  # no se pudo evaluar
    ],
)
def test_decision_de_promocion(new_auc, current_auc, expected):
    assert _decide_promotion(new_auc, current_auc)[0] is expected


def test_un_modelo_mejor_si_reemplaza_al_vigente(client, headers):
    first = _train(client, headers, _signal_csv()).json()
    second = _train(client, headers, _inverted_csv()).json()

    assert second["metrics"]["auc"] > second["metrics"]["current_model_auc"]
    assert second["promoted"] is True
    assert second["current_version_id"] == second["version_id"] != first["version_id"]


def test_la_version_registra_con_que_dataset_se_entreno(client, headers):
    assert _train(client, headers, _signal_csv()).json()["metrics"]["dataset_id"] >= 1


def test_se_puede_volver_a_una_version_anterior(client, headers):
    first = _train(client, headers, _signal_csv()).json()
    second = _train(client, headers, _inverted_csv()).json()

    versions = client.get("/api/v1/lead-match/models", headers=headers).json()
    assert [v["version_id"] for v in versions if v["is_current"]] == [second["version_id"]]
    assert {v["version_id"] for v in versions} == {first["version_id"], second["version_id"]}

    resp = client.post(f"/api/v1/lead-match/models/{first['version_id']}/promote", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["is_current"] is True

    versions = client.get("/api/v1/lead-match/models", headers=headers).json()
    assert [v["version_id"] for v in versions if v["is_current"]] == [first["version_id"]]


def test_promover_una_version_cambia_el_modelo_que_usa_match(client, headers):
    lead = {"lead_id": "l", "origin": "ads", "time_on_site_minutes": 29, "company_size": 10}
    payload = {"leads": [lead], "reps": [{"rep_id": "r", "capacity": 1}]}
    probability = lambda: client.post("/api/v1/lead-match/match", headers=headers, json=payload).json()[  # noqa: E731
        "assignments"
    ][0]["conversion_probability"]

    first = _train(client, headers, _signal_csv()).json()
    assert probability() > 0.5
    _train(client, headers, _inverted_csv())
    assert probability() < 0.5

    client.post(f"/api/v1/lead-match/models/{first['version_id']}/promote", headers=headers)
    assert probability() > 0.5


@pytest.mark.parametrize("version_id", ["no-existe", "..", "..%2F..%2Fotra"])
def test_promover_una_version_inexistente_da_404(client, headers, version_id):
    assert client.post(f"/api/v1/lead-match/models/{version_id}/promote", headers=headers).status_code == 404


def test_no_se_puede_promover_una_version_de_otra_empresa(client, headers):
    version = _train(client, headers, _signal_csv()).json()["version_id"]
    other = client.post("/api/v1/companies", json={"name": "Otra", "contact_email": "otra@empresa.com"})
    other_headers = {"X-API-Key": other.json()["api_key"]}

    assert client.post(f"/api/v1/lead-match/models/{version}/promote", headers=other_headers).status_code == 404


def _rows(row) -> str:
    return HEADER + "".join(row(i) for i in range(40))


@pytest.mark.parametrize(
    ("csv_text", "expected"),
    [
        (_rows(lambda i: f"ads,{i},{i * 3},{'Quizas' if i % 2 else 'No'}\n"), "converted"),
        (_rows(lambda i: f"ads,{i},{i * 3},{(i % 2) * 2}\n"), "converted"),
        (_rows(lambda i: f"ads,{i}min,{i * 3},{i % 2}\n"), "time_on_site_minutes"),
        (_rows(lambda i: f"ads,{i},mucha,{i % 2}\n"), "company_size"),
        (_rows(lambda i: f"ads,inf,{i * 3},{i % 2}\n"), "time_on_site_minutes"),
    ],
)
def test_datos_con_formato_invalido_dan_422_con_la_columna(client, headers, csv_text, expected):
    resp = _train(client, headers, csv_text)
    assert resp.status_code == 422
    assert expected in resp.json()["detail"]


@pytest.mark.parametrize(
    ("positive", "negative"), [("Si", "No"), ("SÍ", "NO"), ("yes", "no"), ("TRUE", "FALSE"), ("1.0", "0.0")]
)
def test_acepta_las_formas_habituales_de_converted(client, headers, positive, negative):
    csv_text = _signal_csv().replace(",1\n", f",{positive}\n").replace(",0\n", f",{negative}\n")
    resp = _train(client, headers, csv_text)
    assert resp.status_code == 200, resp.text
    assert resp.json()["metrics"]["auc"] > 0.9


def test_acepta_coma_decimal_de_excel_en_espanol(client, headers):
    csv_text = _signal_csv().replace(",", ";")
    lines = [line.split(";") for line in csv_text.strip().split("\n")]
    decimal_comma = "\n".join(";".join([a, b.replace(".", ","), c, d]) for a, b, c, d in lines) + "\n"
    assert _train(client, headers, decimal_comma).status_code == 200


def test_predict_many_coincide_con_predict_proba():
    scorer = ConversionScorer(model=None, is_default=True)
    leads = [LeadFeatures(lead_id=f"l{i}", origin="a", time_on_site_minutes=i, company_size=i) for i in range(3)]
    assert scorer.predict_many(leads) == {lead.lead_id: scorer.predict_proba(lead) for lead in leads}
