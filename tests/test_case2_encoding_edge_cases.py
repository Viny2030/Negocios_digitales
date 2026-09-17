"""
Tests de casos borde de app/cases/case2_pricing/encoding.py:
- NaN en columnas numéricas del dataset histórico no debe romper record_to_request/
  request_to_vector, y debe caer al default (mismo criterio que env.py._num).
- Un 0.0 real (inventario agotado, demanda nula) NO debe confundirse con "faltante" y
  reemplazarse por el default -- regresión del bug "or default" original.
- request_to_vector no debe lanzar ZeroDivisionError aunque base_cost/max_multiplier
  vengan en 0 (defensa en profundidad, más allá de la validación de PriceRequest).
"""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from app.cases.case2_pricing.encoding import record_to_request, request_to_vector, row_to_vector
from app.cases.case2_pricing.schemas import PriceRequest


def _base_record(**overrides) -> dict:
    record = {
        "product_id": "sku-1",
        "base_cost": 100.0,
        "user_segment": "standard",
        "inventory_level_pct": 40.0,
        "competitor_avg_price": 130.0,
        "demand_index": 0.6,
        "min_margin_pct": 0.10,
        "max_multiplier": 2.5,
        "min_multiplier": 1.0,
    }
    record.update(overrides)
    return record


def test_record_to_request_nan_en_inventory_cae_al_default():
    record = _base_record(inventory_level_pct=float("nan"))
    request = record_to_request(record)
    assert request.inventory_level_pct == 50.0  # default documentado en record_to_request


def test_record_to_request_nan_en_demand_index_cae_al_default():
    record = _base_record(demand_index=float("nan"))
    request = record_to_request(record)
    assert request.demand_index == 0.5


def test_record_to_request_nan_en_base_cost_cae_al_default():
    record = _base_record(base_cost=float("nan"))
    request = record_to_request(record)
    assert request.base_cost == 1.0


def test_row_to_vector_no_crashea_con_nan_en_varias_columnas():
    record = _base_record(
        inventory_level_pct=float("nan"),
        demand_index=float("nan"),
        base_cost=float("nan"),
        min_margin_pct=float("nan"),
        min_multiplier=float("nan"),
    )
    vector = request_to_vector(record_to_request(record))
    assert vector.shape == (6,)
    assert not np.isnan(vector).any()


def test_inventory_level_pct_cero_real_se_preserva_no_se_reemplaza_por_default():
    record = _base_record(inventory_level_pct=0.0)
    request = record_to_request(record)
    assert request.inventory_level_pct == 0.0

    vector = request_to_vector(request)
    assert vector[0] == 0.0  # inventory_level_pct_norm


def test_demand_index_cero_real_se_preserva_no_se_reemplaza_por_default():
    record = _base_record(demand_index=0.0)
    request = record_to_request(record)
    assert request.demand_index == 0.0

    vector = request_to_vector(request)
    assert vector[5] == 0.0  # demand_index


def test_request_to_vector_no_lanza_zero_division_con_max_multiplier_cero():
    # Bypass deliberado de la validación de PriceRequest (gt=0 en max_multiplier) para
    # ejercer la defensa en profundidad de request_to_vector, que puede recibir datos
    # crudos del histórico (record_to_request) que la validación no cubre en todos los
    # caminos internos.
    request = PriceRequest.model_construct(
        product_id="sku-1",
        base_cost=100.0,
        user_segment="standard",
        inventory_level_pct=50.0,
        competitor_avg_price=130.0,
        demand_index=0.5,
        min_margin_pct=0.10,
        max_multiplier=0.0,
        min_multiplier=1.0,
    )
    vector = request_to_vector(request)
    assert not np.isnan(vector).any()
    assert vector[4] == 1.0  # competitor_norm cae al fallback seguro (1.0) si denom <= 0


def test_request_to_vector_no_lanza_zero_division_con_base_cost_cero():
    request = PriceRequest.model_construct(
        product_id="sku-1",
        base_cost=0.0,
        user_segment="standard",
        inventory_level_pct=50.0,
        competitor_avg_price=130.0,
        demand_index=0.5,
        min_margin_pct=0.10,
        max_multiplier=2.5,
        min_multiplier=1.0,
    )
    vector = request_to_vector(request)
    assert not np.isnan(vector).any()
    assert vector[4] == 1.0


def test_row_to_vector_con_max_multiplier_cero_en_dataset_historico_no_crashea():
    import pandas as pd

    row = pd.Series(_base_record(max_multiplier=0.0))
    # record_to_request valida con PriceRequest (max_multiplier ahora exige > 0), así que
    # una fila histórica corrupta con max_multiplier=0 debe rechazarse de forma explícita
    # en vez de propagar un ZeroDivisionError más abajo.
    with pytest.raises(ValidationError):
        row_to_vector(row)


def test_competitor_avg_price_none_usa_valor_por_defecto_neutral():
    record = _base_record(competitor_avg_price=None)
    request = record_to_request(record)
    vector = request_to_vector(request)
    assert vector[4] == 0.5


def test_record_to_request_nan_en_competitor_avg_price_se_trata_como_ausente():
    record = _base_record(competitor_avg_price=float("nan"))
    request = record_to_request(record)
    assert request.competitor_avg_price is None
    assert not math.isnan(request_to_vector(request)[4])
