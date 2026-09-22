"""
SmartRoute Fleet — Motor de Ruteo y Despacho Last-Mile.

Contrato de negocio: el cliente manda las coordenadas de recogida/entrega, las
ventanas horarias y los vehículos disponibles; la plataforma devuelve la
asignación de pedidos y la secuencia ordenada de paradas por conductor.
Stateless por diseño (a diferencia de dynamic_pricing/lead_match): no hay un
modelo que entrenar ni versionar -- cada request es un problema de optimización
combinatoria resuelto de punta a punta con los datos que trae (ver solver.py).
"""

from pydantic import BaseModel, Field, model_validator


class GeoPoint(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class Stop(BaseModel):
    stop_id: str
    location: GeoPoint
    demand: float = Field(..., ge=0, description="Peso/volumen que ocupa en el vehículo")
    time_window_start_min: int = Field(
        ..., ge=0, description="Minutos desde el inicio del turno en que se abre la ventana"
    )
    time_window_end_min: int = Field(..., description="Minutos desde el inicio del turno en que cierra")
    service_time_min: float = Field(default=5.0, ge=0, description="Minutos que toma la parada en sí")

    @model_validator(mode="after")
    def _ventana_valida(self) -> "Stop":
        if self.time_window_end_min <= self.time_window_start_min:
            raise ValueError(f"stop {self.stop_id}: time_window_end_min debe ser mayor que time_window_start_min")
        return self


class Vehicle(BaseModel):
    vehicle_id: str
    capacity: float = Field(..., gt=0)
    average_speed_kmh: float = Field(default=30.0, gt=0)
    max_route_duration_min: float = Field(default=480.0, gt=0)


class RoutingRequest(BaseModel):
    depot: GeoPoint
    stops: list[Stop] = Field(..., min_length=1)
    vehicles: list[Vehicle] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _ids_unicos(self) -> "RoutingRequest":
        stop_ids = [s.stop_id for s in self.stops]
        if len(stop_ids) != len(set(stop_ids)):
            raise ValueError("stop_id duplicado en stops")
        vehicle_ids = [v.vehicle_id for v in self.vehicles]
        if len(vehicle_ids) != len(set(vehicle_ids)):
            raise ValueError("vehicle_id duplicado en vehicles")
        return self


class RouteStop(BaseModel):
    stop_id: str
    sequence: int
    arrival_min: float = Field(description="Minutos desde el inicio del turno hasta llegar a la parada")
    departure_min: float = Field(description="arrival_min + tiempo de espera por ventana + service_time_min")


class VehicleRoute(BaseModel):
    vehicle_id: str
    stops: list[RouteStop]
    total_distance_km: float
    total_duration_min: float
    load_used: float
    capacity: float


class RoutingResponse(BaseModel):
    routes: list[VehicleRoute]
    unassigned_stops: list[str] = Field(
        default_factory=list,
        description="Paradas que no se pudieron asignar a ningún vehículo (capacidad o "
        "ventana horaria insuficientes con la flota disponible)",
    )
