"""
Entorno Gymnasium para entrenar la política de secuenciación de mensajes (Caso 1.1).
Se usa solo en scripts/train_case1.py (entrenamiento offline), no en el request-path de la API.

El entorno se calibra con el dataset histórico de campañas de la empresa
(exportado de HubSpot/Salesforce/etc., ver app/api/v1/endpoints/datasets.py).
"""

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # gymnasium es opcional hasta que se entrene de verdad
    gym = None
    spaces = None

from app.cases.case1_messaging.encoding import STATE_DIM, _bool, _num, row_to_vector
from app.cases.case1_messaging.schemas import Touchpoint

ACTIONS = list(Touchpoint)

# Peso de cada etapa del funnel en la probabilidad de conversión: un prospecto en
# "decision" convierte más fácil al ser contactado que uno recién en "awareness".
_STAGE_WEIGHT = {"decision": 1.0, "consideration": 0.6, "awareness": 0.3}


class MessagingSequencingEnv(gym.Env if gym else object):
    """
    Estado (vector S): [channels_contacted_onehot(7), messages_received, responses_count,
                         days_since_last_contact, purchase_stage_onehot(3), unsubscribed]
    Acción (A): índice sobre ACTIONS (8 touchpoints, incluyendo "esperar").
    Recompensa (R): +conversión - costo_contacto - penalización_fatiga (si unsubscribe).

    La probabilidad de conversión y de fatiga NO son puramente aleatorias: se calibran
    con las columnas reales del prospecto muestreado (`responses_count`/`messages_received`
    como tasa de respuesta histórica, `purchase_stage` como etapa del funnel,
    `days_since_last_contact` como proxy de qué tan "fresco" es el contacto, y
    `unsubscribed` como señal dura de que ya no hay que volver a contactarlo). Sigue
    siendo un modelo simplificado -- no un simulador real de journeys -- pero a
    diferencia de la versión anterior, el dataset que sube la empresa SÍ cambia lo
    que el agente aprende, no solo la distribución de estados que ve.
    """

    metadata = {"render_modes": []}

    def __init__(self, historical_df=None, max_steps: int = 20):
        super().__init__()
        self.historical_df = historical_df  # DataFrame con journeys históricos reales
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(len(ACTIONS))
        # 7 canales + 2 contadores + 1 recencia + 3 etapas + 1 flag unsubscribed = 14
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32)
        self._step_count = 0
        self._state = None
        self._current_row = None  # fila cruda del prospecto muestreado, para calibrar la recompensa

    def _sample_prospect(self):
        """Calibra el estado inicial/siguiente con un prospecto real del dataset de la
        empresa cuando hay historical_df disponible (mismo mapeo columna->vector que
        usa policy.py en inferencia, ver encoding.py). Si no hay dataset todavía, usa
        un vector de ceros para poder entrenar/probar el entorno igual. Devuelve
        (fila_cruda_o_None, vector_de_estado)."""
        if self.historical_df is not None and len(self.historical_df) > 0:
            row = self.historical_df.sample(1).iloc[0]
            return row, row_to_vector(row)
        return None, np.zeros(STATE_DIM, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._current_row, self._state = self._sample_prospect()
        return self._state, {}

    def _conversion_probability(self) -> float:
        """Probabilidad de conversión al contactar a `self._current_row`, calibrada con
        sus columnas históricas. Sin dataset (`_current_row is None`), cae a un valor
        neutro fijo -- eso es lo que pasaba SIEMPRE antes, incluso con dataset."""
        row = self._current_row
        if row is None:
            return 0.3

        messages_received = _num(row, "messages_received", 0.0)
        responses_count = _num(row, "responses_count", 0.0)
        response_rate = responses_count / messages_received if messages_received > 0 else 0.2

        raw_stage = row.get("purchase_stage", "awareness")
        stage = "awareness" if raw_stage is None or pd.isna(raw_stage) else str(raw_stage)
        stage_weight = _STAGE_WEIGHT.get(stage, 0.4)

        # Recencia: efectividad máxima alrededor de una semana desde el último contacto;
        # ni tan reciente (spam) ni tan lejano (el prospecto ya se enfrió).
        days_since_last_contact = _num(row, "days_since_last_contact", 7.0)
        recency_factor = float(np.clip(1.0 - abs(days_since_last_contact - 7.0) / 21.0, 0.2, 1.0))

        return float(np.clip(0.10 + 0.55 * response_rate + 0.35 * stage_weight * recency_factor, 0.0, 0.95))

    def _fatigue_probability(self) -> float:
        """Probabilidad de que este touchpoint más sea el que provoque un unsubscribe.
        Si el prospecto YA está marcado como unsubscribed en el histórico, contactarlo
        de nuevo es directamente una falta (probabilidad de fatiga = 1, termina el
        episodio): antes eso no se distinguía de un prospecto activo cualquiera."""
        row = self._current_row
        if row is None:
            return 0.02
        if _bool(row, "unsubscribed", False):
            return 1.0
        messages_received = _num(row, "messages_received", 0.0)
        return float(np.clip(0.01 + 0.006 * messages_received, 0.0, 0.2))

    def step(self, action: int):
        self._step_count += 1
        touchpoint = ACTIONS[action]

        if touchpoint == Touchpoint.wait:
            conversion_reward = 0.0
            contact_cost = 0.0
            fatigue_penalty = 0.0
        else:
            converted = np.random.rand() < self._conversion_probability()
            conversion_reward = 1.0 if converted else 0.0
            contact_cost = 0.05
            fatigued = np.random.rand() < self._fatigue_probability()
            fatigue_penalty = -1.0 if fatigued else 0.0

        reward = conversion_reward - contact_cost + fatigue_penalty
        terminated = fatigue_penalty < 0 or self._step_count >= self.max_steps
        truncated = False

        self._current_row, self._state = self._sample_prospect()
        return self._state, reward, terminated, truncated, {}
