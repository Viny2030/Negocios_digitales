"""
Entorno Gymnasium para el agente de pricing dinámico (Caso 2.1). Uso: entrenamiento offline
(scripts/train_case2.py), calibrado con datasets de la empresa (ventas, inventario, costos)
combinados con fuentes públicas de referencia (Inside Airbnb, Expedia) para elasticidad.
"""
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


class PricingEnv(gym.Env if gym else object):
    """
    Estado (S): [inventory_level, marginal_cost_norm, segment_onehot(k), rejection_history,
                 competitor_avg_price_norm, elasticity_estimate]
    Acción (A): multiplicador continuo en [min_multiplier, max_multiplier] sobre el costo base.
    Recompensa (R): revenue_neto - penalización_margen - penalización_equidad (demographic parity).
    """

    metadata = {"render_modes": []}

    def __init__(self, historical_df=None, min_multiplier: float = 1.0, max_multiplier: float = 2.5,
                 fairness_penalty_weight: float = 1.0, max_steps: int = 100):
        super().__init__()
        self.historical_df = historical_df
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.fairness_penalty_weight = fairness_penalty_weight
        self.max_steps = max_steps

        self.action_space = spaces.Box(low=min_multiplier, high=max_multiplier, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self._step_count = 0
        self._state = None
        # Registro de tasas de rechazo por segmento para medir Demographic Parity
        self._rejection_by_segment: dict[str, list[int]] = {}

    def _sample_state(self) -> np.ndarray:
        if self.historical_df is not None and len(self.historical_df) > 0:
            row = self.historical_df.sample(1).iloc[0]
            # TODO: mapear columnas reales (inventario, costo, segmento, competencia) del cliente
        return np.random.uniform(0, 1, size=6).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._state = self._sample_state()
        return self._state, {}

    def _demand_response(self, multiplier: float) -> float:
        """Curva de elasticidad simplificada: a mayor multiplicador, menor probabilidad de compra."""
        return max(0.0, 1.0 - (multiplier - self.min_multiplier) / (self.max_multiplier - self.min_multiplier))

    def step(self, action):
        self._step_count += 1
        multiplier = float(np.clip(action[0], self.min_multiplier, self.max_multiplier))

        purchase_prob = self._demand_response(multiplier)
        purchased = np.random.rand() < purchase_prob
        revenue = multiplier * purchased  # normalizado; en escala real: multiplier * base_cost

        margin_ok = multiplier >= self.min_multiplier * 1.1
        margin_penalty = 0.0 if margin_ok else -1.0

        # Penalización de equidad: si la tasa de rechazo difiere mucho entre segmentos (Demographic Parity)
        fairness_penalty = 0.0  # se calcula de forma agregada en el training loop, ver scripts/train_case2.py

        reward = revenue + margin_penalty + self.fairness_penalty_weight * fairness_penalty
        terminated = self._step_count >= self.max_steps
        truncated = False

        self._state = self._sample_state()
        return self._state, reward, terminated, truncated, {"purchased": bool(purchased), "multiplier": multiplier}
