"""
Tests de app/cases/case2_pricing/fairness.py: el gate de equidad (Demographic Parity)
que decide si un modelo recién entrenado se promueve a producción (ver
scripts/train_case2.py y config.FAIRNESS_DP_THRESHOLD).

Usa un modelo falso (no un PPO/DQN real -- entrenar uno de verdad es lento y no hace
falta para probar la lógica de evaluate_fairness) que favorece deliberadamente a un
segmento sobre otro, para verificar que la métrica efectivamente lo detecta.
"""

import numpy as np
import pandas as pd

from app.cases.case2_pricing.encoding import _stable_segment_code
from app.cases.case2_pricing.fairness import evaluate_fairness

PREMIUM_CODE = _stable_segment_code("premium")
BUDGET_CODE = _stable_segment_code("budget")
_FAVORED_CODE = max(PREMIUM_CODE, BUDGET_CODE)


class _BiasedFakeModel:
    """Le asigna un multiplicador bajo (precio bajo = "resultado favorable") solo al
    segmento con código más alto -- discrimina 100% por segmento, a propósito, para que
    el test tenga una señal inequívoca en vez de depender de un modelo real entrenado."""

    def predict(self, obs, deterministic=True):
        segment_code = float(obs[2])
        multiplier = 1.0 if abs(segment_code - _FAVORED_CODE) < 1e-6 else 2.0
        return np.array([multiplier], dtype=np.float32), None


class _NeutralFakeModel:
    """Mismo multiplicador para todos los segmentos: no debería haber diferencia de DP."""

    def predict(self, obs, deterministic=True):
        return np.array([1.5], dtype=np.float32), None


def _historical_df(n=40, with_segment=True):
    data = {
        "product_id": [f"sku-{i}" for i in range(n)],
        "base_cost": [100.0] * n,
        "inventory_level_pct": [50.0] * n,
        "demand_index": [0.5] * n,
    }
    if with_segment:
        data["user_segment"] = (["premium"] * (n // 2)) + (["budget"] * (n // 2))
    return pd.DataFrame(data)


def test_sin_columna_user_segment_no_bloquea_y_no_evalua():
    df = _historical_df(with_segment=False)
    result = evaluate_fairness(_BiasedFakeModel(), df, min_multiplier=1.0, max_multiplier=2.5)

    assert result["evaluated"] is False
    assert result["demographic_parity_difference"] is None
    assert "user_segment" in result["reason"]


def test_dataset_vacio_no_evalua():
    result = evaluate_fairness(_BiasedFakeModel(), pd.DataFrame(), min_multiplier=1.0, max_multiplier=2.5)
    assert result["evaluated"] is False


def test_sin_modelo_no_evalua():
    df = _historical_df()
    result = evaluate_fairness(None, df, min_multiplier=1.0, max_multiplier=2.5)
    assert result["evaluated"] is False


def test_modelo_sesgado_da_una_diferencia_de_equidad_alta():
    df = _historical_df()
    result = evaluate_fairness(_BiasedFakeModel(), df, min_multiplier=1.0, max_multiplier=2.5)

    assert result["evaluated"] is True
    # Discrimina 100% por segmento -> la diferencia de selection_rate entre segmentos
    # debería ser máxima (cercana a 1.0), muy por encima de settings.FAIRNESS_DP_THRESHOLD (0.10).
    assert result["demographic_parity_difference"] > 0.5
    assert set(result["selection_rate_by_segment"].keys()) == {"premium", "budget"}
    assert result["n_samples"] == len(df)


def test_modelo_neutral_da_una_diferencia_de_equidad_baja():
    df = _historical_df()
    result = evaluate_fairness(_NeutralFakeModel(), df, min_multiplier=1.0, max_multiplier=2.5)

    assert result["evaluated"] is True
    assert result["demographic_parity_difference"] < 0.10  # por debajo del umbral por defecto
