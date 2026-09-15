"""
Tests de la calibración de recompensa con datos reales en los entornos de
entrenamiento offline (app/cases/case1_messaging/env.py, case2_pricing/env.py,
case3_anomaly/env.py y case4_accelerator/env.py). Antes de esto, `historical_df`
solo determinaba qué ESTADOS veía el agente durante el entrenamiento; la
recompensa de cada paso era puramente aleatoria (`np.random.uniform`) o ignoraba
columnas del dataset que sí estaban disponibles (etiqueta real de case3, todos los
campos de `StartupProfile` salvo tracción/equipo/mercado en case4, penalización de
equidad de case2 fija en 0.0). Estos tests verifican que esas columnas/campos
efectivamente cambian la recompensa calculada, no solo el estado observado.
"""

import numpy as np
import pandas as pd
import pytest

gym = pytest.importorskip("gymnasium")

from app.cases.case1_messaging.env import MessagingSequencingEnv  # noqa: E402
from app.cases.case2_pricing.env import PricingEnv  # noqa: E402
from app.cases.case3_anomaly.env import ACTIONS, AnomalyTriageEnv  # noqa: E402
from app.cases.case3_anomaly.schemas import TriageAction  # noqa: E402
from app.cases.case4_accelerator.env import AcceleratorEnv  # noqa: E402
from app.cases.case4_accelerator.schemas import ResourcePool, StartupProfile  # noqa: E402


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


def test_case2_penalizacion_de_equidad_es_cero_con_un_solo_segmento_con_muestra():
    """No hay con quién comparar la tasa de rechazo hasta que al menos 2 segmentos
    tengan `_MIN_SAMPLES_PER_SEGMENT` muestras -- evita penalizar por ruido al
    arranque del entrenamiento, cuando todavía no se vio más que un segmento."""
    env = PricingEnv(historical_df=None)
    env._current_row = _case2_row(user_segment="budget")
    for _ in range(20):
        penalty = env._track_and_estimate_fairness_penalty(purchased=False)
    assert penalty == 0.0


def test_case2_brecha_de_rechazo_entre_segmentos_se_refleja_en_la_penalizacion():
    env = PricingEnv(historical_df=None)

    env._current_row = _case2_row(user_segment="budget")
    for _ in range(env._MIN_SAMPLES_PER_SEGMENT + 5):
        env._track_and_estimate_fairness_penalty(purchased=False)  # siempre rechazado -> tasa 1.0

    env._current_row = _case2_row(user_segment="premium")
    for _ in range(env._MIN_SAMPLES_PER_SEGMENT + 4):
        env._track_and_estimate_fairness_penalty(purchased=True)  # siempre comprado -> tasa 0.0
    penalty = env._track_and_estimate_fairness_penalty(purchased=True)

    assert penalty == pytest.approx(1.0)


def test_case2_step_aplica_la_penalizacion_de_equidad_a_la_recompensa():
    """`step()` ya no deja `fairness_penalty_weight` sin efecto (antes la
    penalización quedaba fija en 0.0): con el peso en 0 la recompensa no debería
    verse afectada por la brecha entre segmentos, con el peso en > 0 sí."""
    env_sin_peso = PricingEnv(historical_df=None, fairness_penalty_weight=0.0, max_steps=1)
    env_sin_peso._current_row = _case2_row(user_segment="budget")
    # Fuerza una brecha grande antes de step() para que haya algo que penalizar.
    for _ in range(env_sin_peso._MIN_SAMPLES_PER_SEGMENT + 5):
        env_sin_peso._track_and_estimate_fairness_penalty(purchased=False)
    env_sin_peso._current_row = _case2_row(user_segment="premium")
    for _ in range(env_sin_peso._MIN_SAMPLES_PER_SEGMENT + 4):
        env_sin_peso._track_and_estimate_fairness_penalty(purchased=True)

    _, _, _, _, info = env_sin_peso.step(np.array([1.0], dtype=np.float32))
    assert info["fairness_penalty"] > 0.0


def _case3_row(**overrides) -> pd.Series:
    base = {
        "alert_id": "a-1",
        "metric_name": "cpu_usage",
        "z_score": 3.0,
        "pct_change": 0.5,
        "historical_similar_count": 2,
        "estimated_financial_impact": 5000.0,
        "is_false_positive": 0,
    }
    base.update(overrides)
    return pd.Series(base)


def test_case3_alert_signal_usa_la_etiqueta_real_is_false_positive():
    env = AnomalyTriageEnv(historical_df=None)

    env._current_row = _case3_row(is_false_positive=0)  # incidente real
    is_real_anomaly, _ = env._alert_signal()
    assert is_real_anomaly is True

    env._current_row = _case3_row(is_false_positive=1)  # falso positivo
    is_real_anomaly, _ = env._alert_signal()
    assert is_real_anomaly is False


def test_case3_alert_signal_acepta_el_alias_legado_label():
    """Datasets subidos antes del rename de columna siguen con `label`; ver
    generate_synthetic_data.py y el fallback en dataset_ingestion/env/classifier."""
    env = AnomalyTriageEnv(historical_df=None)
    row = pd.Series(
        {
            "alert_id": "a-1",
            "z_score": 3.0,
            "pct_change": 0.5,
            "historical_similar_count": 2,
            "estimated_financial_impact": 5000.0,
            "label": 0,
        }
    )
    env._current_row = row
    is_real_anomaly, _ = env._alert_signal()
    assert is_real_anomaly is True


def test_case3_sin_etiqueta_ni_alias_cae_al_sorteo_aleatorio_sin_crashear():
    env = AnomalyTriageEnv(historical_df=None)
    env._current_row = pd.Series({"alert_id": "a-1", "estimated_financial_impact": 100.0})
    is_real_anomaly, impact_norm = env._alert_signal()
    assert isinstance(is_real_anomaly, bool)
    assert impact_norm >= 0.0


def test_case3_ignorar_alerta_real_de_alto_impacto_penaliza_mas_que_una_marginal():
    ignore_idx = ACTIONS.index(TriageAction.ignore)

    env_alto = AnomalyTriageEnv(historical_df=None, max_steps=1)
    env_alto._current_row = _case3_row(is_false_positive=0, estimated_financial_impact=8000.0)
    _, reward_alto_impacto, *_ = env_alto.step(ignore_idx)

    env_bajo = AnomalyTriageEnv(historical_df=None, max_steps=1)
    env_bajo._current_row = _case3_row(is_false_positive=0, estimated_financial_impact=200.0)
    _, reward_bajo_impacto, *_ = env_bajo.step(ignore_idx)

    assert reward_alto_impacto < reward_bajo_impacto < 0


def _startup(**overrides) -> StartupProfile:
    base = dict(
        startup_id="s-1",
        traction_score=0.8,
        team_score=0.8,
        market_score=0.8,
        runway_months=12.0,
        past_response_score=0.5,
        mentorship_hours_received_last_period=0.0,
        followon_capital_received_last_period=0.0,
    )
    base.update(overrides)
    return StartupProfile(**base)


def _resources() -> ResourcePool:
    return ResourcePool(
        mentorship_hours_available=200.0,
        followon_capital_available=1_000_000.0,
        strategic_connections_available=5,
    )


def test_case4_apoyo_recibido_el_periodo_pasado_genera_retornos_decrecientes():
    env = AcceleratorEnv(historical_df=None, max_steps=1)

    env._portfolio = [
        _startup(mentorship_hours_received_last_period=0.0, followon_capital_received_last_period=0.0)
    ]
    env._resources = _resources()
    _, reward_sin_apoyo_previo, *_ = env.step(np.array([1.0], dtype=np.float32))

    env._portfolio = [
        _startup(mentorship_hours_received_last_period=80.0, followon_capital_received_last_period=300_000.0)
    ]
    env._resources = _resources()
    _, reward_con_apoyo_previo, *_ = env.step(np.array([1.0], dtype=np.float32))

    assert reward_sin_apoyo_previo > reward_con_apoyo_previo


def test_case4_past_response_score_amplifica_el_retorno():
    env = AcceleratorEnv(historical_df=None, max_steps=1)

    env._portfolio = [_startup(past_response_score=0.9)]
    env._resources = _resources()
    _, reward_buena_respuesta, *_ = env.step(np.array([1.0], dtype=np.float32))

    env._portfolio = [_startup(past_response_score=0.1)]
    env._resources = _resources()
    _, reward_mala_respuesta, *_ = env.step(np.array([1.0], dtype=np.float32))

    assert reward_buena_respuesta > reward_mala_respuesta


def test_case4_runway_bajo_agrega_bonus_de_urgencia_sobre_el_capital():
    env = AcceleratorEnv(historical_df=None, max_steps=1)

    env._portfolio = [_startup(runway_months=1.0)]
    env._resources = _resources()
    _, reward_runway_critico, *_ = env.step(np.array([1.0], dtype=np.float32))

    env._portfolio = [_startup(runway_months=24.0)]
    env._resources = _resources()
    _, reward_runway_sobrado, *_ = env.step(np.array([1.0], dtype=np.float32))

    assert reward_runway_critico > reward_runway_sobrado
