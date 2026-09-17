"""
Entorno Gymnasium para el agente de pricing dinámico (Caso 2.1). Uso: entrenamiento offline
(scripts/train_case2.py), calibrado con datasets de la empresa (ventas, inventario, costos)
combinados con fuentes públicas de referencia (Inside Airbnb, Expedia) para elasticidad.
"""

from collections import deque

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

from app.cases.case2_pricing.encoding import STATE_DIM, _num, row_to_vector

# Elasticidad relativa por segmento: budget cae más rápido al subir el multiplicador
# (más sensible al precio), premium cae más lento. Multiplica el término de precio de
# _demand_response -- no es un dato del paper, es un placeholder razonable para que el
# entrenamiento tenga una señal de "quién es más sensible al precio" por segmento, ver
# también el docstring de la clase.
_SEGMENT_ELASTICITY = {"premium": 0.5, "standard": 1.0, "budget": 1.5}


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

    La penalización de equidad YA se aplica dentro del loop de entrenamiento (antes
    quedaba fija en 0.0 acá, con un comentario diciendo "se calcula de forma agregada
    en el training loop" que en realidad no pasaba en ningún lado): `step()` mantiene
    una ventana móvil de tasa de rechazo (no-compra) por segmento y penaliza según la
    brecha entre el segmento más y el menos rechazado, ver `_track_and_estimate_fairness_penalty`.
    Es una señal barata y online -- NO reemplaza a `evaluate_fairness` (Fairlearn,
    Demographic Parity real sobre todo el `historical_df`), que sigue siendo el gate
    de aprobación después de entrenar (ver `scripts/train_case2.py`); esto solo le da
    a la política una señal de equidad DURANTE el entrenamiento, no solo al final.
    """

    metadata = {"render_modes": []}

    # Tamaño de la ventana móvil de rechazos por segmento (ver _track_and_estimate_fairness_penalty)
    # y mínimo de muestras por segmento antes de confiar en su tasa de rechazo estimada.
    _FAIRNESS_WINDOW = 200
    _MIN_SAMPLES_PER_SEGMENT = 5

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
        # Ventana móvil de resultados (1 = rechazo/no-compra, 0 = compra) por segmento,
        # para estimar una Demographic Parity "en vivo" durante el entrenamiento.
        self._rejection_by_segment: dict[str, deque] = {}

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

    def _track_and_estimate_fairness_penalty(self, purchased: bool) -> float:
        """Actualiza la ventana móvil de rechazo del segmento de `self._current_row` y
        devuelve una estimación barata de Demographic Parity: la brecha entre la tasa
        de rechazo (no-compra) del segmento más castigado y la del menos castigado,
        entre los segmentos que ya tienen suficientes muestras (`_MIN_SAMPLES_PER_SEGMENT`)
        en la ventana. 0.0 mientras no haya al menos 2 segmentos con muestra suficiente
        -- evita penalizar por ruido al arranque del entrenamiento, antes de que haya
        datos de más de un segmento."""
        row = self._current_row
        segment = str(row.get("user_segment") or "sin_segmento") if row is not None else "sin_segmento"

        bucket = self._rejection_by_segment.setdefault(segment, deque(maxlen=self._FAIRNESS_WINDOW))
        bucket.append(0 if purchased else 1)

        ready = [b for b in self._rejection_by_segment.values() if len(b) >= self._MIN_SAMPLES_PER_SEGMENT]
        if len(ready) < 2:
            return 0.0

        rejection_rates = [sum(b) / len(b) for b in ready]
        return float(max(rejection_rates) - min(rejection_rates))

    def step(self, action):
        self._step_count += 1
        multiplier = float(np.clip(action[0], self.min_multiplier, self.max_multiplier))

        purchase_prob = self._demand_response(multiplier)
        purchased = np.random.rand() < purchase_prob
        revenue = multiplier * purchased  # normalizado; en escala real: multiplier * base_cost

        margin_ok = multiplier >= self.min_multiplier * 1.1
        margin_penalty = 0.0 if margin_ok else -1.0

        # Penalización de equidad: brecha de rechazo entre segmentos en la ventana móvil
        # (Demographic Parity "en vivo" -- ver docstring de la clase y del método de arriba).
        fairness_penalty = self._track_and_estimate_fairness_penalty(purchased)

        reward = revenue + margin_penalty - self.fairness_penalty_weight * fairness_penalty
        terminated = self._step_count >= self.max_steps
        truncated = False

        self._current_row, self._state = self._sample_state()
        return (
            self._state,
            reward,
            terminated,
            truncated,
            {"purchased": bool(purchased), "multiplier": multiplier, "fairness_penalty": fairness_penalty},
        )
