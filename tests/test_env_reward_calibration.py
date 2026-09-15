"""
Tests de la calibración de recompensa con datos reales en los entornos de
entrenamiento offline (app/cases/case1_messaging/env.py y
app/cases/case2_pricing/env.py). Antes de esto, `historical_df` solo determinaba
qué ESTADOS veía el agente durante el entrenamiento; la recompensa de cada paso
era puramente aleatoria (`np.random.uniform`), sin usar ninguna columna real del
dataset subido por la empresa. Estos tests verifican que las columnas históricas
(`responses_count`, `purchase_stage`, `unsubscribed`, `demand_index`,
`user_segment`, `competitor_avg_price`) efectivamente cambian la probabilidad
calculada, no solo el estado observado.
"""

import numpy as np
import pandas as pd
import pytest

gym = pytest.importorskip("gymnasium")

from app.cases.case1_messaging.env import MessagingSequencingEnv  # noqa: E402
from app.cases.case2_pricing.env import PricingEnv  # noqa: E402


def _case1_row(**overrides) -> pd.Series:
    base = {
        "prospect_id": "p-1",
        "channels_contacted": "",
        "messages_received": 5,
        "responses_count": 0,
        "days_since_last_contact": 7.0,
        "purchase_stage": "awareness",
        "unsubscribed": False,
    }
    base.update(overrides)
    return pd.Series(base)


def test_case1_prospecto_en_decision_con_buena_tasa_de_respuesta_convierte_mas():
    env = MessagingSequencingEnv(historical_df=None)

    env._current_row = _case1_row(purchase_stage="decision", messages_received=4, responses_count=3)
    prob_alta = env._conversion_probability()

    env._current_row = _case1_row(purchase_stage="awareness", messages_received=10, responses_count=0)
    prob_baja = env._conversion_probability()

    assert prob_alta > prob_baja


def test_case1_prospecto_ya_unsubscribed_tiene_fatiga_maxima():
    env = MessagingSequencingEnv(historical_df=None)
    env._current_row = _case1_row(unsubscribed=True)
    assert env._fatigue_probability() == 1.0


def test_case1_nan_en_columnas_numericas_no_crashea_y_cae_al_default():
    """Una empresa real casi seguro va a tener NaN en algunas filas -- antes
    `row.get(key, default) or default` no atrapaba NaN (es 'truthy' en Python)."""
    env = MessagingSequencingEnv(historical_df=None)
    env._current_row = _case1_row(messages_received=np.nan, responses_count=np.nan, days_since_last_contact=np.nan)
    prob = env._conversion_probability()
    assert 0.0 <= prob <= 1.0


def test_case1_sin_dataset_usa_default_neutro():
    env = MessagingSequencingEnv(historical_df=None)
    env._current_row = None
    assert env._conversion_probability() == pytest.approx(0.3)
    assert env._fatigue_probability() == pytest.approx(0.02)


def _case2_row(**overrides) -> pd.Series:
    base = {
        "product_id": "sku-1",
        "base_cost": 100.0,
        "user_segment": "standard",
        "inventory_level_pct": 50.0,
        "competitor_avg_price": None,
        "demand_index": 0.5,
    }
    base.update(overrides)
    return pd.Series(base)


def test_case2_segmento_budget_cae_mas_rapido_que_premium_al_subir_precio():
    env = PricingEnv(historical_df=None, min_multiplier=1.0, max_multiplier=2.0)
    multiplier_alto = 1.8

    env._current_row = _case2_row(user_segment="budget")
    prob_budget = env._demand_response(multiplier_alto)

    env._current_row = _case2_row(user_segment="premium")
    prob_premium = env._demand_response(multiplier_alto)

    assert prob_premium > prob_budget


def test_case2_mayor_demand_index_aumenta_probabilidad_de_compra():
    env = PricingEnv(historical_df=None, min_multiplier=1.0, max_multiplier=2.0)

    env._current_row = _case2_row(demand_index=0.1)
    prob_baja_demanda = env._demand_response(1.5)

    env._current_row = _case2_row(demand_index=0.9)
    prob_alta_demanda = env._demand_response(1.5)

    assert prob_alta_demanda > prob_baja_demanda


def test_case2_precio_muy_por_encima_del_competidor_penaliza():
    env = PricingEnv(historical_df=None, min_multiplier=1.0, max_multiplier=2.0)

    env._current_row = _case2_row(competitor_avg_price=100.0)  # mismo precio que el propio (multiplier=1.0 * 100)
    prob_a_la_par = env._demand_response(1.0)

    env._current_row = _case2_row(competitor_avg_price=60.0)  # muy por debajo -- quedamos caros en comparación
    prob_caro_vs_competencia = env._demand_response(1.0)

    assert prob_a_la_par > prob_caro_vs_competencia


def test_case2_nan_en_competitor_avg_price_no_crashea():
    env = PricingEnv(historical_df=None, min_multiplier=1.0, max_multiplier=2.0)
    env._current_row = _case2_row(competitor_avg_price=np.nan, demand_index=np.nan)
    prob = env._demand_response(1.5)
    assert 0.0 <= prob <= 1.0


def test_case2_sin_fila_usa_curva_simple_por_multiplicador():
    env = PricingEnv(historical_df=None, min_multiplier=1.0, max_multiplier=2.0)
    env._current_row = None
    assert env._demand_response(1.0) == pytest.approx(1.0)
    assert env._demand_response(2.0) == pytest.approx(0.0)
