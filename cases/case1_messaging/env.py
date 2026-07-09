"""
Entorno Gymnasium para entrenar la política de secuenciación de mensajes (Caso 1.1).
Se usa solo en scripts/train_case1.py (entrenamiento offline), no en el request-path de la API.

El entorno se calibra con el dataset histórico de campañas de la empresa
(exportado de HubSpot/Salesforce/etc., ver app/api/v1/endpoints/datasets.py).
"""
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # gymnasium es opcional hasta que se entrene de verdad
    gym = None
    spaces = None

from app.cases.case1_messaging.schemas import Touchpoint

ACTIONS = list(Touchpoint)


class MessagingSequencingEnv(gym.Env if gym else object):
    """
    Estado (vector S): [channels_contacted_onehot(7), messages_received, responses_count,
                         days_since_last_contact, purchase_stage_onehot(3), unsubscribed]
    Acción (A): índice sobre ACTIONS (8 touchpoints, incluyendo "esperar").
    Recompensa (R): +conversión - costo_contacto - penalización_fatiga (si unsubscribe).
    """

    metadata = {"render_modes": []}

    def __init__(self, historical_df=None, max_steps: int = 20):
        super().__init__()
        self.historical_df = historical_df  # DataFrame con journeys históricos reales
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(len(ACTIONS))
        # 7 canales + 2 contadores + 1 recencia + 3 etapas + 1 flag unsubscribed = 14
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(14,), dtype=np.float32)
        self._step_count = 0
        self._state = None

    def _sample_prospect(self) -> np.ndarray:
        if self.historical_df is not None and len(self.historical_df) > 0:
            row = self.historical_df.sample(1).iloc[0]
            # TODO: mapear columnas reales del CRM del cliente a este vector de estado
        return np.zeros(14, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._state = self._sample_prospect()
        return self._state, {}

    def step(self, action: int):
        self._step_count += 1
        touchpoint = ACTIONS[action]

        # --- Placeholder de dinámica / recompensa ---
        # En producción esto se calibra con datos históricos reales (respuesta observada
        # a touchpoints similares) o con un simulador de journeys (ver README del caso).
        conversion_reward = np.random.uniform(0, 1) if touchpoint != Touchpoint.wait else 0.0
        contact_cost = 0.0 if touchpoint == Touchpoint.wait else 0.05
        fatigue_penalty = -1.0 if np.random.rand() < 0.02 and touchpoint != Touchpoint.wait else 0.0

        reward = conversion_reward - contact_cost + fatigue_penalty
        terminated = fatigue_penalty < 0 or self._step_count >= self.max_steps
        truncated = False

        self._state = self._sample_prospect()
        return self._state, reward, terminated, truncated, {}
