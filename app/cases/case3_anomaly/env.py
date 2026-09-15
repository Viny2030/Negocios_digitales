"""
Entorno Gymnasium para el agente de triaje de anomalías (Caso 3.3). Entrenamiento offline
(scripts/train_case3.py) calibrado con NAB / Yahoo Anomaly Dataset + logs de tickets propios.
"""

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

from app.cases.case3_anomaly.encoding import STATE_DIM, row_to_vector
from app.cases.case3_anomaly.schemas import TriageAction

ACTIONS = list(TriageAction)

# Fracción del impacto estimado que se "recupera" según qué tan a fondo se investigó
# la alerta -- una acción liviana (5 min) recupera menos que escalar a ingeniería.
_EFFECTIVENESS = {
    TriageAction.ignore: 0.0,
    TriageAction.investigate_light: 0.4,
    TriageAction.investigate_deep: 0.9,
    TriageAction.escalate_engineering: 0.85,
    TriageAction.escalate_business: 0.75,
    TriageAction.merge_related: 0.5,
}


def _num(row: pd.Series, key: str, default: float) -> float:
    """Ver mismo helper en app/cases/case1_messaging/env.py: `row.get(key, default)
    or default` no atrapa NaN porque un NaN de pandas es "truthy" en Python."""
    value = row.get(key, default)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return float(value)


class AnomalyTriageEnv(gym.Env if gym else object):
    """
    Estado (S): [z_score_norm, tiempo_transcurrido_norm, frecuencia_mensual_norm,
                 horas_analista_disponibles_norm, alertas_pendientes_norm, impacto_estimado_norm]
    Acción (A): índice sobre ACTIONS (6 acciones de triaje).
    Recompensa (R): impacto recuperado - costo_tiempo_investigacion - penalización_falso_positivo
                    - penalización_severa_por_ignorar_alerta_real.

    Si la alerta muestreada trae la columna `is_false_positive` (o su alias legado
    `label`: 1 = falso positivo, 0 = incidente real -- ver
    app/cases/case3_anomaly/classifier.py), esa etiqueta histórica real decide si la
    alerta era o no una anomalía real, en vez de sortearlo con una moneda al 30%
    como antes (que ignoraba por completo la etiqueta, aunque estuviera en el
    dataset). El impacto "recuperado" al investigar también sale de
    `estimated_financial_impact` de la fila, no de un rango aleatorio fijo -- así que
    ignorar una alerta real de alto impacto ahora penaliza más que ignorar una
    marginal. Sin esas columnas (dataset sin etiquetar todavía), cae al sorteo
    aleatorio anterior para poder entrenar/probar igual.
    """

    metadata = {"render_modes": []}

    def __init__(self, historical_df=None, max_steps: int = 200):
        super().__init__()
        self.historical_df = historical_df
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32)
        self._step_count = 0
        self._state = None
        self._current_row = None  # fila cruda de la alerta muestreada, para calibrar la recompensa

    def _sample_alert(self):
        """Calibra el estado con una alerta real del dataset (NAB / Yahoo Anomaly
        Dataset / tickets propios); mismo mapeo columna->vector que policy.py en
        inferencia (ver encoding.py). La capacidad del equipo se simula porque no es
        un atributo de la alerta histórica. Sin dataset todavía, cae a un estado
        aleatorio para poder entrenar/probar el entorno igual. Devuelve
        (fila_cruda_o_None, vector_de_estado)."""
        if self.historical_df is not None and len(self.historical_df) > 0:
            row = self.historical_df.sample(1).iloc[0]
            return row, row_to_vector(row)
        return None, np.random.uniform(0, 1, size=STATE_DIM).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._current_row, self._state = self._sample_alert()
        return self._state, {}

    _INVESTIGATION_COST = {
        TriageAction.ignore: 0.0,
        TriageAction.investigate_light: 0.05,
        TriageAction.investigate_deep: 0.2,
        TriageAction.escalate_engineering: 0.1,
        TriageAction.escalate_business: 0.1,
        TriageAction.merge_related: 0.02,
    }

    def _alert_signal(self) -> tuple[bool, float]:
        """(es_anomalía_real, impacto_estimado_normalizado) de `self._current_row`.
        Prioriza la etiqueta histórica real (`is_false_positive`/`label`) sobre el
        sorteo aleatorio; ídem con `estimated_financial_impact` para el impacto."""
        row = self._current_row
        if row is None:
            return bool(np.random.rand() < 0.3), float(np.random.uniform(0.5, 1.5))

        label_value = row.get("is_false_positive", row.get("label"))
        is_real_anomaly = None
        if label_value is not None:
            try:
                if not pd.isna(label_value):
                    is_real_anomaly = int(label_value) == 0  # 1 = false_positive, 0 = real_incident
            except (TypeError, ValueError):
                is_real_anomaly = None
        if is_real_anomaly is None:
            is_real_anomaly = bool(np.random.rand() < 0.3)

        impact = _num(row, "estimated_financial_impact", 0.0)
        # Normalizado contra 5000 (orden de magnitud típico de un incidente real en el
        # dataset sintético, ver generate_synthetic_data.py): ~1.0 para un incidente
        # promedio, más para uno grande. Acotado para que un outlier no vuele la escala.
        impact_norm = float(np.clip(impact / 5000.0, 0.0, 3.0))
        return is_real_anomaly, impact_norm

    def step(self, action: int):
        self._step_count += 1
        chosen = ACTIONS[action]
        is_real_anomaly, impact_norm = self._alert_signal()

        cost = self._INVESTIGATION_COST[chosen]
        if is_real_anomaly and chosen == TriageAction.ignore:
            # Peor cuanto más impacto tenía la alerta real que se ignoró (antes era un
            # -2.0 fijo sin importar si la alerta era marginal o crítica).
            reward = -2.0 * max(impact_norm, 0.2)
        elif not is_real_anomaly and chosen in (
            TriageAction.investigate_deep,
            TriageAction.escalate_engineering,
            TriageAction.escalate_business,
        ):
            reward = -cost  # costo hundido en falso positivo investigado a fondo
        elif is_real_anomaly and chosen != TriageAction.ignore:
            recovered_impact = _EFFECTIVENESS[chosen] * impact_norm
            reward = recovered_impact - cost
        else:
            reward = -cost * 0.2  # ignorar ruido real: correcto, costo mínimo

        terminated = self._step_count >= self.max_steps
        truncated = False
        self._current_row, self._state = self._sample_alert()
        return self._state, reward, terminated, truncated, {"is_real_anomaly": bool(is_real_anomaly)}
