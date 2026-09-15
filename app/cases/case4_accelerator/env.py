"""
Entorno Gymnasium para el agente de asignación de recursos en la aceleradora (Caso
6.2). A diferencia de los otros 3 casos, un "step" representa la decisión sobre TODO
el portfolio del período (no una decisión aislada sobre un prospecto/precio/alerta):
el agente devuelve un score de valor esperado por cada slot de startup, el entorno
resuelve la asignación vía el mismo LP de producción (constraints.py) y calcula el
retorno simulado del portfolio para ese período.

Uso: entrenamiento offline (scripts/train_case4.py), calibrado con datasets de
portfolios históricos de aceleradoras (Crunchbase, YC/500 Startups open data,
Endeavor LatAm) o con portfolios sintéticos heterogéneos.
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

from app.cases.case4_accelerator.constraints import apply_portfolio_allocation
from app.cases.case4_accelerator.encoding import (
    MAX_STARTUPS,
    STATE_DIM,
    portfolio_to_vector,
    sample_portfolio_from_df,
)
from app.cases.case4_accelerator.schemas import ResourcePool


class AcceleratorEnv(gym.Env if gym else object):
    """
    Estado (S): portfolio de hasta MAX_STARTUPS startups (tracción, equipo, mercado,
    runway, respuesta a intervenciones previas, apoyo recibido) + contexto de
    recursos disponibles del período.
    Acción (A): vector de scores de valor esperado en [0,1], uno por slot de
    startup — el LP de constraints.py traduce esos scores en la asignación real de
    recursos, igual que en producción (service.py).
    Recompensa (R): retorno simulado del portfolio neto del costo de recursos
    asignados. La dinámica de retorno de acá abajo sigue siendo un placeholder --
    calibrarla con retorno real observado (exits, follow-on rounds, supervivencia a
    18 meses) es trabajo de datos que depende del dataset histórico real de cada
    aceleradora -- pero ya usa TODAS las columnas del `StartupProfile`, no solo
    tracción/equipo/mercado: `past_response_score` (qué tan bien historicamente
    aprovechó el apoyo recibido) amplifica o atenúa el retorno; el apoyo YA recibido
    el período pasado (`mentorship_hours_received_last_period`,
    `followon_capital_received_last_period`) genera retornos decrecientes sobre el
    apoyo nuevo (a un startup que ya venía muy apoyado, lo mismo de nuevo le rinde
    menos); y un `runway_months` bajo agrega un pequeño bonus al capital asignado
    (ese capital puede ser la diferencia entre sobrevivir o no).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        historical_df=None,
        max_steps: int = 50,
        default_mentorship_hours: float = 160.0,
        default_followon_capital: float = 200_000.0,
        default_strategic_connections: int = 5,
    ):
        super().__init__()
        self.historical_df = historical_df
        self.max_steps = max_steps
        self._default_resources = ResourcePool(
            mentorship_hours_available=default_mentorship_hours,
            followon_capital_available=default_followon_capital,
            strategic_connections_available=default_strategic_connections,
        )

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(MAX_STARTUPS,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32)

        self._step_count = 0
        self._portfolio = []
        self._resources = self._default_resources
        self._state = None

    def _sample_portfolio_and_resources(self):
        if self.historical_df is not None and len(self.historical_df) > 0:
            portfolio = sample_portfolio_from_df(self.historical_df, n=MAX_STARTUPS)
        else:
            portfolio = []
        return portfolio, self._default_resources

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._portfolio, self._resources = self._sample_portfolio_and_resources()
        self._state = portfolio_to_vector(self._portfolio, self._resources)
        return self._state, {}

    def step(self, action: np.ndarray):
        self._step_count += 1

        value_scores = {
            startup.startup_id: float(np.clip(score, 0.0, 1.0)) for startup, score in zip(self._portfolio, action)
        }
        allocations, _ = apply_portfolio_allocation(value_scores, self._portfolio, self._resources)

        # --- Placeholder de dinámica de retorno del portfolio ---
        # Función creciente y cóncava del recurso recibido, ponderada por los
        # fundamentals de cada startup: refleja que concentrar recursos en las
        # startups con mejores fundamentals rinde más que repartir parejo, que es el
        # punto de negocio central del caso, sin pretender ser el modelo real de
        # retorno (eso requiere datos reales de exits/supervivencia).
        portfolio_return = 0.0
        resource_cost = 0.0
        for startup, allocation in zip(self._portfolio, allocations):
            fundamentals = 0.4 * startup.traction_score + 0.3 * startup.team_score + 0.3 * startup.market_score
            resource_intensity = np.tanh(
                allocation.mentorship_hours / 20.0 + allocation.followon_capital_recommended / 50_000.0
            )

            # Retornos decrecientes: si YA recibió mucho apoyo el período pasado, el
            # mismo monto ahora rinde menos (evita que el LP concentre todo siempre en
            # el mismo startup período tras período sin límite).
            prior_support = np.tanh(
                startup.mentorship_hours_received_last_period / 40.0
                + startup.followon_capital_received_last_period / 100_000.0
            )
            saturation = float(np.clip(1.0 - 0.5 * prior_support, 0.5, 1.0))

            # Qué tan bien este startup aprovechó históricamente el apoyo recibido:
            # amplifica el retorno de los que respondieron bien, lo atenúa (no lo anula)
            # en los que no.
            responsiveness = 0.5 + 0.5 * startup.past_response_score

            portfolio_return += fundamentals * resource_intensity * responsiveness * saturation

            # Urgencia de runway: con poco runway, el capital asignado AHORA vale más
            # (puede ser la diferencia entre sobrevivir o no) -- bonus chico y acotado,
            # solo sobre el capital, no sobre mentoría.
            runway_urgency = float(np.clip(1.0 - startup.runway_months / 12.0, 0.0, 0.5))
            portfolio_return += runway_urgency * min(allocation.followon_capital_recommended / 50_000.0, 1.0) * 0.3

            resource_cost += (
                0.01 * allocation.mentorship_hours + 0.000005 * allocation.followon_capital_recommended
            )

        reward = portfolio_return - resource_cost
        terminated = self._step_count >= self.max_steps
        truncated = False

        self._portfolio, self._resources = self._sample_portfolio_and_resources()
        self._state = portfolio_to_vector(self._portfolio, self._resources)
        return self._state, reward, terminated, truncated, {"portfolio_return": portfolio_return}
