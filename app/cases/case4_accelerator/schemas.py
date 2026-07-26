"""
Caso 6.2 — Política de asignación de recursos en incubadora o aceleradora.

Estado (S): estado de cada startup del portfolio (tracción, equipo, mercado, runway,
apoyo recibido, respuesta a intervenciones anteriores) + recursos disponibles de la
incubadora en el período (horas de mentoría, capital follow-on, conexiones activables).
Acciones (A): asignación de recursos entre startups del portfolio para el próximo mes
(horas de mentoría, recomendación de capital follow-on, activación de conexión
estratégica).
Recompensa (R): retorno del portfolio (exits, follow-on rounds levantadas,
supervivencia a 18 meses) neto del costo de los recursos asignados.

Arquitectura de este caso (distinta a los otros 3, ver docstring de env.py): en vez
de que el agente elija una acción discreta o un precio continuo, decide sobre un
portfolio completo. Se separa en dos capas: la política (RL o heurística) estima un
score de valor esperado por startup dado su estado; el módulo LP (constraints.py)
resuelve la asignación óptima del presupuesto de la incubadora sobre esos scores,
sujeta a capacidad total y a un tope opcional de concentración por startup. Esto es
fiel al enfoque del enunciado: "RL con múltiples agentes pasivos (las startups) y un
agente activo (la incubadora); función de recompensa definida sobre el portfolio
completo".
"""
from pydantic import BaseModel, Field


class StartupProfile(BaseModel):
    """Estado (S) de una startup del portfolio."""

    startup_id: str
    traction_score: float = Field(..., ge=0, le=1, description="Tracción normalizada (revenue, usuarios, crecimiento)")
    team_score: float = Field(..., ge=0, le=1)
    market_score: float = Field(..., ge=0, le=1, description="Tamaño y momentum del mercado objetivo")
    runway_months: float = Field(..., ge=0)
    past_response_score: float = Field(
        default=0.5, ge=0, le=1,
        description="Qué tan bien respondió esta startup a intervenciones previas (0=nula, 1=excelente)",
    )
    mentorship_hours_received_last_period: float = Field(default=0.0, ge=0)
    followon_capital_received_last_period: float = Field(default=0.0, ge=0)


class ResourcePool(BaseModel):
    """Recursos disponibles de la aceleradora para el período (mes) a asignar."""

    mentorship_hours_available: float = Field(..., gt=0)
    followon_capital_available: float = Field(default=0.0, ge=0)
    strategic_connections_available: int = Field(default=0, ge=0)

    # Topes de concentración: por defecto sin límite (el enunciado señala que
    # concentrar recursos suele superar a la distribución uniforme), pero
    # configurables si la aceleradora quiere forzar diversificación de riesgo.
    max_mentorship_share_per_startup: float = Field(default=1.0, gt=0, le=1)
    max_capital_share_per_startup: float = Field(default=1.0, gt=0, le=1)


class StartupAllocation(BaseModel):
    startup_id: str
    mentorship_hours: float
    followon_capital_recommended: float
    strategic_connection_activated: bool
    expected_value_score: float = Field(
        description="Score de valor esperado estimado por la política, antes del filtro LP",
    )


class AllocationRequest(BaseModel):
    startups: list[StartupProfile]
    resources: ResourcePool


class AllocationResponse(BaseModel):
    allocations: list[StartupAllocation]
    constraints_applied: list[str] = Field(default_factory=list)
    rationale: str


class TrainRequest(BaseModel):
    dataset_id: int
    algorithm: str = Field(
        default="ppo",
        description="Único soportado: el espacio de acciones es continuo (Box), DQN no aplica.",
    )
    timesteps: int = 50_000