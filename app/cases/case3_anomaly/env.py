"""
Entorno Gymnasium para el agente de triaje de anomalías (Caso 3.3). Entrenamiento offline
(scripts/train_case3.py) calibrado con NAB / Yahoo Anomaly Dataset + logs de tickets propios.
"""
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

from app.cases.case3_anomaly.schemas import TriageAction

ACTIONS = list(TriageAction)


class AnomalyTriageEnv(gym.Env if gym else object):
    """
    Estado (S): [z_score_norm, tiempo_transcurrido_norm, frecuencia_mensual_norm,
                 horas_analista_disponibles_norm, alertas_pendientes_norm, impacto_estimado_norm]
    Acción (A): índice sobre ACTIONS (6 acciones de triaje).
    Recompensa (R): impacto recuperado - costo_tiempo_investigacion - penalización_falso_positivo
                    - penalización_severa_por_ignorar_alerta_real.
    """

    metadata = {"render_modes": []}

    def __init__(self, historical_df=None, max_steps: int = 200):
        super().__init__()
        self.historical_df = historical_df
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self._step_count = 0
        self._state = None

    def _sample_alert(self) -> np.ndarray:
        if self.historical_df is not None and len(self.historical_df) > 0:
            row = self.historical_df.sample(1).iloc[0]
            # TODO: mapear columnas reales del dataset (NAB / tickets internos) a este vector
        return np.random.uniform(0, 1, size=6).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._state = self._sample_alert()
        return self._state, {}

    _INVESTIGATION_COST = {
        TriageAction.ignore: 0.0,
        TriageAction.investigate_light: 0.05,
        TriageAction.investigate_deep: 0.2,
        TriageAction.escalate_engineering: 0.1,
        TriageAction.escalate_business: 0.1,
        TriageAction.merge_related: 0.02,
    }

    def step(self, action: int):
        self._step_count += 1
        chosen = ACTIONS[action]
        is_real_anomaly = np.random.rand() < 0.3  # placeholder; en training real viene etiquetado

        cost = self._INVESTIGATION_COST[chosen]
        if is_real_anomaly and chosen == TriageAction.ignore:
            reward = -2.0  # penalización severa por ignorar una anomalía real
        elif not is_real_anomaly and chosen in (
            TriageAction.investigate_deep, TriageAction.escalate_engineering, TriageAction.escalate_business,
        ):
            reward = -cost  # costo hundido en falso positivo investigado a fondo
        elif is_real_anomaly and chosen != TriageAction.ignore:
            recovered_impact = np.random.uniform(0.5, 1.5)
            reward = recovered_impact - cost
        else:
            reward = -cost * 0.2  # ignorar ruido real: correcto, costo mínimo

        terminated = self._step_count >= self.max_steps
        truncated = False
        self._state = self._sample_alert()
        return self._state, reward, terminated, truncated, {"is_real_anomaly": bool(is_real_anomaly)}
