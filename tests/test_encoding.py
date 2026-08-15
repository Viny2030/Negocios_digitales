"""
Prueba que los env.py de los 3 casos ya calibran el estado con datos históricos reales
en vez de descartarlos (el bug que se corrigió: `_sample_*` muestreaba una fila del
DataFrame y la tiraba, devolviendo ruido aleatorio o ceros).
"""

import numpy as np
import pandas as pd

from app.cases.case1_messaging.encoding import (
    CHANNELS as CASE1_CHANNELS,
)
from app.cases.case1_messaging.encoding import (
    PURCHASE_STAGES,
)
from app.cases.case1_messaging.encoding import (
    row_to_vector as case1_row_to_vector,
)
from app.cases.case2_pricing.encoding import row_to_vector as case2_row_to_vector
from app.cases.case3_anomaly.encoding import row_to_vector as case3_row_to_vector


def test_case1_row_to_vector_reflects_real_columns():
    row = pd.Series(
        {
            "prospect_id": "p-1",
            "channels_contacted": "email_a;sms",
            "messages_received": 3,
            "responses_count": 1,
            "days_since_last_contact": 10,
            "purchase_stage": "consideration",
            "unsubscribed": False,
        }
    )
    vector = case1_row_to_vector(row)

    assert vector.shape == (14,)
    assert vector[CASE1_CHANNELS.index("email_a")] == 1.0
    assert vector[CASE1_CHANNELS.index("sms")] == 1.0
    assert vector[CASE1_CHANNELS.index("call")] == 0.0  # canal no contactado
    assert np.isclose(vector[7], 0.3)  # messages_received / 10
    assert np.isclose(vector[8], 0.2)  # responses_count / 5
    assert np.isclose(vector[9], 10 / 30)  # days_since_last_contact / 30
    stage_offset = 10
    assert vector[stage_offset + PURCHASE_STAGES.index("consideration")] == 1.0
    assert vector[-1] == 0.0  # unsubscribed


def test_case1_row_to_vector_differs_between_rows():
    """Antes del fix, cualquier fila producía un vector de ceros: dos prospectos
    distintos deben producir vectores distintos."""
    row_a = pd.Series(
        {
            "prospect_id": "p-a",
            "channels_contacted": "email_a",
            "messages_received": 1,
            "responses_count": 0,
            "days_since_last_contact": 1,
            "purchase_stage": "awareness",
            "unsubscribed": False,
        }
    )
    row_b = pd.Series(
        {
            "prospect_id": "p-b",
            "channels_contacted": "call;sms",
            "messages_received": 8,
            "responses_count": 4,
            "days_since_last_contact": 25,
            "purchase_stage": "decision",
            "unsubscribed": False,
        }
    )
    assert not np.allclose(case1_row_to_vector(row_a), case1_row_to_vector(row_b))


def test_case2_row_to_vector_reflects_real_columns():
    row = pd.Series(
        {
            "product_id": "sku-1",
            "base_cost": 50.0,
            "user_segment": "premium",
            "inventory_level_pct": 80.0,
            "competitor_avg_price": 120.0,
            "demand_index": 0.7,
        }
    )
    vector = case2_row_to_vector(row)

    assert vector.shape == (6,)
    assert np.isclose(vector[0], 0.8)  # inventory_level_pct / 100
    assert np.isclose(vector[1], 0.05)  # base_cost / 1000
    assert np.isclose(vector[5], 0.7)  # demand_index


def test_case2_row_to_vector_handles_missing_competitor_price():
    row = pd.Series(
        {
            "product_id": "sku-2",
            "base_cost": 30.0,
            "user_segment": "standard",
            "inventory_level_pct": 40.0,
            "competitor_avg_price": float("nan"),
            "demand_index": 0.3,
        }
    )
    vector = case2_row_to_vector(row)
    assert vector.shape == (6,)
    assert np.isclose(vector[4], 0.5)  # default competitor_norm cuando no hay dato


def test_case3_row_to_vector_reflects_real_columns():
    row = pd.Series(
        {
            "alert_id": "a-1",
            "metric_name": "checkout_conversion",
            "z_score": 4.0,
            "pct_change": -10.0,
            "triggered_at": "2026-01-01T00:00:00",
            "historical_similar_count": 5,
            "estimated_financial_impact": 6000.0,
        }
    )
    vector = case3_row_to_vector(row)

    assert vector.shape == (6,)
    assert np.isclose(vector[0], 0.8)  # z_score / 5, capado a 1.0
    assert np.isclose(vector[2], 0.25)  # historical_similar_count / 20
    assert np.isclose(vector[5], 0.6)  # estimated_financial_impact / 10000
    # índices 1 (tiempo transcurrido, placeholder) y 3-4 (capacidad simulada) están
    # fuera del control de la fila histórica; solo se valida que estén en rango.
    assert 0.0 <= vector[3] <= 1.0
    assert 0.0 <= vector[4] <= 1.0
