"""Tests de SmartRoute Fleet: cálculo de distancias, armado de rutas por Clarke-Wright,
capacidad, ventanas horarias y el endpoint end-to-end."""

from app.cases.smart_route.schemas import GeoPoint, RoutingRequest, Stop, Vehicle
from app.cases.smart_route.solver import haversine_km, solve

DEPOT = GeoPoint(lat=-34.6037, lon=-58.3816)  # Obelisco, CABA


def _stop(stop_id, lat, lon, demand=1.0, start=0, end=600, service=5.0) -> Stop:
    return Stop(
        stop_id=stop_id,
        location=GeoPoint(lat=lat, lon=lon),
        demand=demand,
        time_window_start_min=start,
        time_window_end_min=end,
        service_time_min=service,
    )


def test_haversine_km_punto_consigo_mismo_es_cero():
    assert haversine_km(DEPOT, DEPOT) == 0.0


def test_haversine_km_distancia_conocida_aproximada():
    # 1 grado de latitud equivale a ~111.19 km en cualquier punto del planeta
    # (es la definición del radio terrestre usado en la fórmula) -- un chequeo
    # matemático verificable, no depende de memoria geográfica de la ciudad.
    un_grado_al_sur = GeoPoint(lat=DEPOT.lat - 1.0, lon=DEPOT.lon)
    dist = haversine_km(DEPOT, un_grado_al_sur)
    assert 110.5 < dist < 111.5


def test_todas_las_paradas_asignadas_con_capacidad_y_ventanas_amplias():
    stops = [
        _stop("s1", -34.60, -58.38),
        _stop("s2", -34.61, -58.39),
        _stop("s3", -34.59, -58.37),
        _stop("s4", -34.62, -58.40),
    ]
    vehicles = [Vehicle(vehicle_id="v1", capacity=100, average_speed_kmh=25)]
    request = RoutingRequest(depot=DEPOT, stops=stops, vehicles=vehicles)

    response = solve(request)

    assigned_ids = {rs.stop_id for r in response.routes for rs in r.stops}
    assert assigned_ids == {"s1", "s2", "s3", "s4"}
    assert response.unassigned_stops == []
    assert response.routes[0].vehicle_id == "v1"
    # La secuencia debe estar numerada desde 0 sin huecos.
    sequences = sorted(rs.sequence for rs in response.routes[0].stops)
    assert sequences == list(range(len(response.routes[0].stops)))


def test_capacidad_reparte_entre_dos_vehiculos():
    # 4 paradas de demanda 6 c/u = 24 en total; un solo vehículo de capacidad 10 no
    # alcanza, hacen falta al menos dos.
    stops = [_stop(f"s{i}", -34.60 + i * 0.01, -58.38 + i * 0.01, demand=6.0) for i in range(4)]
    vehicles = [
        Vehicle(vehicle_id="v1", capacity=10, average_speed_kmh=25),
        Vehicle(vehicle_id="v2", capacity=20, average_speed_kmh=25),
    ]
    request = RoutingRequest(depot=DEPOT, stops=stops, vehicles=vehicles)

    response = solve(request)

    assigned_ids = {rs.stop_id for r in response.routes for rs in r.stops}
    assert assigned_ids == {s.stop_id for s in stops}
    assert response.unassigned_stops == []
    # Ningún vehículo debe superar su propia capacidad.
    for route in response.routes:
        assert route.load_used <= route.capacity + 1e-9


def test_capacidad_total_insuficiente_deja_paradas_sin_asignar():
    stops = [_stop(f"s{i}", -34.60 + i * 0.01, -58.38, demand=100.0) for i in range(3)]
    vehicles = [Vehicle(vehicle_id="v1", capacity=50, average_speed_kmh=25)]
    request = RoutingRequest(depot=DEPOT, stops=stops, vehicles=vehicles)

    response = solve(request)

    # Ninguna parada de demanda 100 entra en un vehículo de capacidad 50 -- todas
    # quedan sin asignar, en vez de que el solver reviente o mienta la carga.
    assert set(response.unassigned_stops) == {s.stop_id for s in stops}
    assert response.routes == []


def test_ventana_horaria_imposible_queda_sin_asignar_en_vez_de_mentir_el_horario():
    # Un solo vehículo lento, y una parada con una ventana horaria que cierra antes
    # de que el vehículo pueda llegar físicamente (está lejísimos y la ventana es
    # angosta al principio del turno).
    imposible = _stop("lejos", -35.5, -59.5, start=0, end=5)
    cercana = _stop("cerca", -34.605, -58.382, start=0, end=600)
    vehicles = [Vehicle(vehicle_id="v1", capacity=100, average_speed_kmh=20)]
    request = RoutingRequest(depot=DEPOT, stops=[imposible, cercana], vehicles=vehicles)

    response = solve(request)

    assert "lejos" in response.unassigned_stops
    # La parada alcanzable sigue asignada -- no se descarta la ruta entera por una
    # sola parada inviable.
    assigned_ids = {rs.stop_id for r in response.routes for rs in r.stops}
    assert "cerca" in assigned_ids


def test_endpoint_plan_routes(client, company_api_key):
    payload = {
        "depot": {"lat": -34.6037, "lon": -58.3816},
        "stops": [
            {
                "stop_id": "a",
                "location": {"lat": -34.60, "lon": -58.38},
                "demand": 2,
                "time_window_start_min": 0,
                "time_window_end_min": 300,
            },
            {
                "stop_id": "b",
                "location": {"lat": -34.61, "lon": -58.39},
                "demand": 3,
                "time_window_start_min": 0,
                "time_window_end_min": 300,
            },
        ],
        "vehicles": [{"vehicle_id": "v1", "capacity": 20, "average_speed_kmh": 25}],
    }
    resp = client.post("/api/v1/smart-route/plan-routes", json=payload, headers={"X-API-Key": company_api_key})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["routes"]) == 1
    assert body["unassigned_stops"] == []


def test_endpoint_plan_routes_requiere_api_key(client):
    payload = {
        "depot": {"lat": 0, "lon": 0},
        "stops": [
            {
                "stop_id": "a",
                "location": {"lat": 0.01, "lon": 0.01},
                "demand": 1,
                "time_window_start_min": 0,
                "time_window_end_min": 100,
            }
        ],
        "vehicles": [{"vehicle_id": "v1", "capacity": 10}],
    }
    resp = client.post("/api/v1/smart-route/plan-routes", json=payload)
    assert resp.status_code == 401


def test_stop_con_ventana_invertida_es_rechazada_por_schema():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _stop("mala", -34.6, -58.4, start=100, end=50)
