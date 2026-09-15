"""
Entorno Gymnasium para el agente de pricing dinámico (Caso 2.1). Uso: entrenamiento offline
(scripts/train_case2.py), calibrado con datasets de la empresa (ventas, inventario, costos)
combinados con fuentes públicas de referencia (Inside Airbnb, Expedia) para elasticidad.
"""

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

from app.cases.case2_pricing.encoding import STATE_DIM, row_to_vector

# Elasticidad relativa por segmento: budget cae más rápido al subir el multiplicador
# (más sensible al precio), premium cae más lento. Multiplica el término de precio de
# _demand_response -- no es un dato del paper, es un placeholder razonable para que el
# entrenamiento tenga una señal de "quién es más sensible al precio" por segmento, ver
# también el docstring de la clase.
_SEGMENT_ELASTICITY = {"premium": 0.5, "standard": 1.0, "budget": 1.5}


def _num(row: pd.Series, key: str, default: float | None) -> float | None:
    """Lee una columna numérica de la fila tolerando NaN/None -- ver mismo helper en
    app/cases/case1_messaging/env.py. Puede devolver `default=None` para distinguir
    "la columna no vino/estaba vacía" de "vino con un 0 real"."""
    value = row.get(key, default)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return float(value)


class PricingEnv(gym.Env if gym else object):
    """
    Estado (S): [inventory_level, marginal_cost_norm, segment_onehot(k), rejection_history,
                 competitor_avg_price_norm, elasticity_estimate]
    Acción (A): multiplicador continuo en [min_multiplier, max_multiplier] sobre el costo base.
    Recompensa (R): revenue_neto - penalización_margen - penalización_equidad (demographic parity).

    La probabilidad de compra ya no depende solo del multiplicador elegido: se calibra
    con `demand_index`, `competitor_avg_price` y `user_segment` de la transacción
    muestreada del dataset histórico (antes esas columnas se codificaban en el estado
    pero no afectaban en nada la dinámica de compra simulada -- el dataset que subía la
    empresa cambiaba qué veía el agente, no qué aprendía de eso).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        historical_df=None,
        min_multiplier: float = 1.0,
        max_multiplier: float = 2.5,
        fairness_penalty_weight: float = 1.0,
        max_steps: int = 100,
    ):
        super().__init__()
        self.historical_df = historical_df
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.fairness_penalty_weight = fairness_penalty_weight
        self.max_steps = max_steps

        self.action_space = spaces.Box(low=min_multiplier, high=max_multiplier, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32)
        self._step_count = 0
        self._state = None
        self._current_row = None  # fila cruda de la transacción muestreada, para calibrar la demanda
        # Registro de tasas de rechazo por segmento para medir Demographic Parity
        self._rejection_by_segment: dict[str, list[int]] = {}

    def _sample_state(self):
        """Calibra el estado con una transacción real del dataset de la empresa (mismo
        mapeo columna->vector que usa policy.py en inferencia, ver encoding.py). Sin
        dataset todavía, cae a un estado aleatorio para poder entrenar/probar igual.
        Devuelve (fila_cruda_o_None, vector_de_estado)."""
        if self.historical_df is not None and len(self.historical_df) > 0:
            row = self.historical_df.sample(1).iloc[0]
            return row, row_to_vector(row)
        return None, np.random.uniform(0, 1, size=STATE_DIM).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._current_row, self._state = self._sample_state()
        return self._state, {}

    def _demand_response(self, multiplier: float) -> float:
        """Probabilidad de compra a `multiplier`, calibrada con la fila muestreada:
        - `demand_index` (0-1 del dataset): más demanda, más probabilidad de compra a
          cualquier precio.
        - `user_segment`: cada segmento tiene una elasticidad distinta (ver
          _SEGMENT_ELASTICITY) -- budget cae más rápido al subir el precio.
        - `competitor_avg_price` (si viene en el dataset): si nuestro precio queda muy
          por encima del promedio de mercado, penaliza más allá de la caída "normal"
          por elasticidad; si queda por debajo, la favorece.
        Sin fila (dataset vacío), cae a la curva simple anterior (solo por multiplicador)."""
        row = self._current_row
        base_range = max(self.max_multiplier - self.min_multiplier, 1e-6)
        price_position = (multiplier - self.min_multiplier) / base_range

        if row is None:
            return max(0.0, 1.0 - price_position)

        segment = str(row.get("user_segment") or "standard")
        elasticity = _SEGMENT_ELASTICITY.get(segment, 1.0)
        price_effect = max(0.0, 1.0 - elasticity * price_position)

        demand_index = _num(row, "demand_index", 0.5)
        demand_factor = 0.4 + 0.6 * demand_index  # entre 0.4 (sin demanda) y 1.0 (demanda máxima)

        competitive_factor = 1.0
        base_cost = _num(row, "base_cost", 0.0)
        competitor_avg_price = _num(row, "competitor_avg_price", None)
        if base_cost > 0 and competitor_avg_price is not None and competitor_avg_price > 0:
            own_price = multiplier * base_cost
            relative_price = own_price / competitor_avg_price
            competitive_factor = float(np.clip(1.4 - relative_price * 0.5, 0.5, 1.3))

        return float(np.clip(price_effect * demand_factor * competitive_factor, 0.0, 1.0))

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

        self._current_row, self._state = self._sample_state()
        return self._state, reward, terminated, truncated, {"purchased": bool(purchased), "multiplier": multiplier}
