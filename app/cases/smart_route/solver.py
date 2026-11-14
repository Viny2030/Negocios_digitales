"""
VRPTW (Vehicle Routing Problem with Time Windows) resuelto con el algoritmo de
ahorros de Clarke-Wright + una mejora local 2-opt, en vez de fuerza bruta (que
explota factorialmente: con 15 paradas ya son ~1.3e12 permutaciones posibles).

El proceso tiene 4 fases separadas, cada una con su propia responsabilidad:

  1. `_build_routes_by_savings`: arma las rutas por distancia y capacidad
     (Clarke-Wright clásico), ignorando ventanas horarias -- esa fase es sobre
     TODO el pool de paradas, antes de saber qué vehículo específico maneja
     cada ruta, así que no hay una velocidad de vehículo concreta todavía con
     la cual calcular horarios reales.
  2. `_assign_routes_to_vehicles`: empareja cada ruta construida con un
     vehículo real (greedy por capacidad), y ahí recién se conoce la
     velocidad con la que calcular tiempos de viaje.
  3. `_make_feasible`: con el vehículo ya asignado, valida el horario real
     parada por parada. Si una parada llega tarde a su ventana (o la ruta se
     pasa de `max_route_duration_min`), intenta un 2-opt acotado para
     reordenar sin violar ventanas; si ni así entra, saca la parada más
     conflictiva de la ruta en vez de devolver un horario inviable.
  4. `_insert_unassigned`: las fases anteriores no conocen las ventanas al
     armar las rutas, así que pueden dejar afuera paradas que sí entraban.
     Acá cada parada suelta prueba todas las posiciones de todos los vehículos
     (también los que quedaron libres) y va a la que menos distancia agrega.
     Solo las que no entran en ninguno quedan en `unassigned_stops`.

La respuesta siempre cumple capacidad, ventanas y duración según este modelo, que
usa distancia en línea recta y velocidad constante: sin calles ni tránsito, los
tiempos reales serán mayores. Es una heurística: no garantiza el óptimo global.
"""

import math
from dataclasses import dataclass, field

from app.cases.smart_route.schemas import (
    GeoPoint,
    RouteStop,
    RoutingRequest,
    RoutingResponse,
    Stop,
    Vehicle,
    VehicleRoute,
)

EARTH_RADIUS_KM = 6371.0


def haversine_km(a: GeoPoint, b: GeoPoint) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [a.lat, a.lon, b.lat, b.lon])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


@dataclass
class _Route:
    stops: list[Stop] = field(default_factory=list)

    @property
    def demand(self) -> float:
        return sum(s.demand for s in self.stops)


def _build_routes_by_savings(depot: GeoPoint, stops: list[Stop], max_capacity: float) -> list[_Route]:
    """Clarke-Wright: arranca con una ruta por parada y va fusionando los pares
    con mayor ahorro (s(i,j) = d(depot,i) + d(depot,j) - d(i,j)) mientras el
    resultado siga siendo un camino válido (solo se pueden fusionar por los
    EXTREMOS de cada ruta, nunca por el medio) y no supere `max_capacity`."""
    routes: list[_Route] = [_Route(stops=[s]) for s in stops]
    route_of: dict[str, _Route] = {s.stop_id: r for s, r in zip(stops, routes, strict=True)}

    dist_depot = {s.stop_id: haversine_km(depot, s.location) for s in stops}
    dist = {}
    for i in stops:
        for j in stops:
            if i.stop_id != j.stop_id:
                dist[(i.stop_id, j.stop_id)] = haversine_km(i.location, j.location)

    savings = []
    for i in stops:
        for j in stops:
            if i.stop_id >= j.stop_id:
                continue
            s = dist_depot[i.stop_id] + dist_depot[j.stop_id] - dist[(i.stop_id, j.stop_id)]
            savings.append((s, i.stop_id, j.stop_id))
    savings.sort(key=lambda t: t[0], reverse=True)

    for _, i_id, j_id in savings:
        route_i = route_of[i_id]
        route_j = route_of[j_id]
        if route_i is route_j:
            continue
        if route_i.demand + route_j.demand > max_capacity:
            continue

        i_is_last = route_i.stops[-1].stop_id == i_id
        i_is_first = route_i.stops[0].stop_id == i_id
        j_is_last = route_j.stops[-1].stop_id == j_id
        j_is_first = route_j.stops[0].stop_id == j_id

        # Solo se puede fusionar si ambos puntos de unión son extremos de su
        # ruta -- si alguno está en el medio, fusionar ahí rompería el camino
        # (dejaría una parada con dos "próximas paradas" en simultáneo).
        if i_is_last and j_is_first:
            merged = route_i.stops + route_j.stops
        elif i_is_first and j_is_last:
            merged = route_j.stops + route_i.stops
        elif i_is_last and j_is_last:
            merged = route_i.stops + list(reversed(route_j.stops))
        elif i_is_first and j_is_first:
            merged = list(reversed(route_i.stops)) + route_j.stops
        else:
            continue  # ambos son interiores de su ruta -- no fusionable

        new_route = _Route(stops=merged)
        for s in merged:
            route_of[s.stop_id] = new_route
        # Reemplaza las dos rutas viejas por la nueva en la lista de trabajo.
        routes = [r for r in routes if r is not route_i and r is not route_j]
        routes.append(new_route)

    return routes


def _assign_routes_to_vehicles(
    routes: list[_Route], vehicles: list[Vehicle]
) -> tuple[list[tuple[_Route, Vehicle]], list[str]]:
    """Empareja cada ruta con el vehículo más chico que todavía le entra (greedy
    por capacidad), para no desperdiciar vehículos grandes en rutas livianas.
    Si sobran rutas sin vehículo disponible que las cubra, sus paradas quedan
    sin asignar en vez de forzar una asignación que exceda capacidad."""
    routes_sorted = sorted(routes, key=lambda r: r.demand, reverse=True)
    vehicles_sorted = sorted(vehicles, key=lambda v: v.capacity)

    assigned: list[tuple[_Route, Vehicle]] = []
    unassigned_stop_ids: list[str] = []
    used_vehicle_ids: set[str] = set()

    for route in routes_sorted:
        candidate = next(
            (v for v in vehicles_sorted if v.vehicle_id not in used_vehicle_ids and v.capacity >= route.demand),
            None,
        )
        if candidate is None:
            unassigned_stop_ids.extend(s.stop_id for s in route.stops)
            continue
        used_vehicle_ids.add(candidate.vehicle_id)
        assigned.append((route, candidate))

    return assigned, unassigned_stop_ids


def _schedule(depot: GeoPoint, ordered_stops: list[Stop], speed_kmh: float) -> list[dict]:
    """Simula el recorrido en el orden dado y devuelve, por parada, el punto de
    llegada/salida en minutos desde el inicio del turno -- incluye la espera si
    el vehículo llega antes de que abra la ventana horaria."""
    schedule = []
    current_time = 0.0
    current_point = depot
    for stop in ordered_stops:
        dist_km = haversine_km(current_point, stop.location)
        travel_min = (dist_km / speed_kmh) * 60
        arrival = current_time + travel_min
        start_service = max(arrival, stop.time_window_start_min)
        departure = start_service + stop.service_time_min
        schedule.append(
            {
                "stop": stop,
                "arrival_min": arrival,
                "departure_min": departure,
                "on_time": arrival <= stop.time_window_end_min,
            }
        )
        current_time = departure
        current_point = stop.location
    return schedule


def _route_distance_km(depot: GeoPoint, ordered_stops: list[Stop]) -> float:
    total = 0.0
    current = depot
    for stop in ordered_stops:
        total += haversine_km(current, stop.location)
        current = stop.location
    total += haversine_km(current, depot)
    return total


def _route_duration_min(depot: GeoPoint, schedule: list[dict], speed_kmh: float) -> float:
    """Duración total de la ruta: hasta que el vehículo vuelve al depósito."""
    if not schedule:
        return 0.0
    last = schedule[-1]
    return last["departure_min"] + (haversine_km(last["stop"].location, depot) / speed_kmh) * 60


def _is_feasible(depot: GeoPoint, ordered_stops: list[Stop], vehicle: Vehicle) -> bool:
    schedule = _schedule(depot, ordered_stops, vehicle.average_speed_kmh)
    if not all(row["on_time"] for row in schedule):
        return False
    return _route_duration_min(depot, schedule, vehicle.average_speed_kmh) <= vehicle.max_route_duration_min


def _two_opt_improve(depot: GeoPoint, stops: list[Stop], speed_kmh: float) -> list[Stop]:
    """2-opt acotado: prueba invertir cada segmento [i:j] y se queda con la
    mejora si (a) reduce la distancia total Y (b) el horario resultante no
    empeora ninguna parada que antes llegaba a horario. No persigue el óptimo
    global (eso requeriría muchas más iteraciones/reinicios) -- alcanza para
    limpiar los cruces obvios que el merge por ahorros puede dejar."""
    best = list(stops)
    best_distance = _route_distance_km(depot, best)
    improved = True
    while improved and len(best) >= 3:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                candidate = best[:i] + list(reversed(best[i : j + 1])) + best[j + 1 :]
                candidate_distance = _route_distance_km(depot, candidate)
                if candidate_distance >= best_distance - 1e-9:
                    continue
                candidate_schedule = _schedule(depot, candidate, speed_kmh)
                if all(row["on_time"] for row in candidate_schedule):
                    best = candidate
                    best_distance = candidate_distance
                    improved = True
                    break
            if improved:
                break
    return best


def _make_feasible(depot: GeoPoint, stops: list[Stop], vehicle: Vehicle) -> tuple[list[Stop], list[str]]:
    """Aplica 2-opt y, si todavía queda alguna parada tarde o la ruta se pasa
    de `max_route_duration_min`, saca la parada más tarde una por una hasta
    que el resto sea factible. Devuelve (paradas_factibles, ids_descartados)."""
    ordered = _two_opt_improve(depot, stops, vehicle.average_speed_kmh)
    dropped: list[str] = []

    while ordered:
        schedule = _schedule(depot, ordered, vehicle.average_speed_kmh)
        total_duration = _route_duration_min(depot, schedule, vehicle.average_speed_kmh)
        late_rows = [row for row in schedule if not row["on_time"]]
        over_duration = total_duration > vehicle.max_route_duration_min

        if not late_rows and not over_duration:
            return ordered, dropped

        if late_rows:
            worst = max(late_rows, key=lambda row: row["arrival_min"] - row["stop"].time_window_end_min)
        else:
            worst = schedule[-1]  # se pasó de duración total: sacamos la última parada

        dropped.append(worst["stop"].stop_id)
        ordered = [s for s in ordered if s.stop_id != worst["stop"].stop_id]

    return ordered, dropped


def _insert_unassigned(
    depot: GeoPoint,
    plans: dict[str, list[Stop]],
    vehicles: list[Vehicle],
    unassigned: list[Stop],
) -> list[str]:
    """Inserta cada parada suelta donde menos distancia agregue sin romper capacidad,
    ventanas ni duración. Modifica `plans` y devuelve los ids que no entraron en ningún
    vehículo. Primero van las de ventana más temprana, que tienen menos lugares posibles.
    Los vehículos se recorren de menor a mayor capacidad: ante un empate, usa el más chico."""
    vehicles_by_capacity = sorted(vehicles, key=lambda v: v.capacity)
    remaining: list[str] = []

    for stop in sorted(unassigned, key=lambda s: (s.time_window_end_min, s.stop_id)):
        best: tuple[float, str, list[Stop]] | None = None
        for vehicle in vehicles_by_capacity:
            current = plans.get(vehicle.vehicle_id, [])
            if sum(s.demand for s in current) + stop.demand > vehicle.capacity:
                continue
            current_distance = _route_distance_km(depot, current)
            for position in range(len(current) + 1):
                candidate = current[:position] + [stop] + current[position:]
                if not _is_feasible(depot, candidate, vehicle):
                    continue
                added = _route_distance_km(depot, candidate) - current_distance
                if best is None or added < best[0] - 1e-9:
                    best = (added, vehicle.vehicle_id, candidate)

        if best is None:
            remaining.append(stop.stop_id)
        else:
            plans[best[1]] = best[2]

    return remaining


def solve(request: RoutingRequest) -> RoutingResponse:
    max_capacity = max(v.capacity for v in request.vehicles)
    raw_routes = _build_routes_by_savings(request.depot, request.stops, max_capacity)
    assigned, unassigned_ids = _assign_routes_to_vehicles(raw_routes, request.vehicles)

    plans: dict[str, list[Stop]] = {}
    for route, vehicle in assigned:
        feasible_stops, dropped = _make_feasible(request.depot, route.stops, vehicle)
        unassigned_ids.extend(dropped)
        if feasible_stops:
            plans[vehicle.vehicle_id] = feasible_stops

    stops_by_id = {s.stop_id: s for s in request.stops}
    remaining = _insert_unassigned(
        request.depot, plans, request.vehicles, [stops_by_id[stop_id] for stop_id in unassigned_ids]
    )

    vehicles_by_id = {v.vehicle_id: v for v in request.vehicles}
    final_routes: list[VehicleRoute] = []
    for vehicle_id, feasible_stops in plans.items():
        vehicle = vehicles_by_id[vehicle_id]
        schedule = _schedule(request.depot, feasible_stops, vehicle.average_speed_kmh)
        route_stops = [
            RouteStop(
                stop_id=row["stop"].stop_id,
                sequence=idx,
                arrival_min=round(row["arrival_min"], 2),
                departure_min=round(row["departure_min"], 2),
            )
            for idx, row in enumerate(schedule)
        ]
        final_routes.append(
            VehicleRoute(
                vehicle_id=vehicle.vehicle_id,
                stops=route_stops,
                total_distance_km=round(_route_distance_km(request.depot, feasible_stops), 3),
                total_duration_min=round(
                    _route_duration_min(request.depot, schedule, vehicle.average_speed_kmh), 2
                ),
                load_used=sum(s.demand for s in feasible_stops),
                capacity=vehicle.capacity,
            )
        )

    return RoutingResponse(routes=final_routes, unassigned_stops=sorted(remaining))
