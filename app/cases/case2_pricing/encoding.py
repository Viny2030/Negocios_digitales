"""
Codificación de estado del Caso 2.1 — única fuente de verdad, usada tanto por
`policy.py` (inferencia) como por `env.py` (entrenamiento offline con historical_df).
Ver app/cases/case1_messaging/encoding.py para la misma razón de ser: entrenar y
predecir con codificaciones distintas invalida el modelo.
"""

import hashlib

import numpy as np
import pandas as pd

from app.cases.case2_pricing.schemas import PriceRequest

STATE_DIM = 6


def _num(row, key: str, default: float | None) -> float | None:
    """Lee un campo numérico de una fila/registro tolerando NaN/None -- `row.get(key,
    default) or default` NO alcanza acá porque un NaN de pandas es "truthy" en Python
    (no es falsy como 0 o None), así que se colaba tal cual en vez de caer al default;
    además, `or default` trataría un 0.0 real (ej. inventario agotado, demanda nula)
    como "faltante" y lo reemplazaría por el default. Puede devolver `default=None`
    para distinguir "la columna no vino/estaba vacía" de "vino con un 0 real". Misma
    fuente de verdad usada por `env.py` (ver también app/cases/case1_messaging/env.py)."""
    value = row.get(key, default)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _stable_segment_code(segment: str) -> float:
    """Codifica `user_segment` a un número en [0, 1). Usa hashlib en vez del `hash()`
    builtin de Python: `hash()` de un string está salado por proceso (PYTHONHASHSEED,
    aleatorio por defecto desde Python 3.3), así que el mismo segmento producía un
    código distinto en cada reinicio del proceso -- y, peor, distinto entre el proceso
    de la API y el worker de Celery que entrena. hashlib.sha256 es determinista entre
    procesos y entre corridas, así que el mismo segmento siempre cae en el mismo bucket.
    Sigue siendo un placeholder grosero (colisiones entre segmentos distintos son
    posibles); para producción, reemplazar por un vocabulario explícito de segmentos
    conocidos de la empresa (ver record_to_request) con un índice fijo por segmento."""
    digest = hashlib.sha256(segment.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 100) / 100.0


def request_to_vector(request: PriceRequest) -> np.ndarray:
    """Vector de estado (S): [inventory_level_pct_norm, base_cost_norm, segment_encoded,
    rejection_history(placeholder), competitor_price_norm, demand_index]."""
    competitor_norm = 0.5
    if request.competitor_avg_price is not None:
        denom = request.base_cost * request.max_multiplier
        competitor_norm = min(request.competitor_avg_price / denom, 1.0) if denom > 0 else 1.0
    vector = [
        request.inventory_level_pct / 100.0,
        min(request.base_cost / 1000.0, 1.0),
        _stable_segment_code(request.user_segment),
        0.0,  # historial de rechazo (vendría del dataset/CRM)
        competitor_norm,
        request.demand_index,
    ]
    return np.array(vector, dtype=np.float32)


def record_to_request(record: dict) -> PriceRequest:
    """Mapea una fila del dataset histórico de la empresa (transacciones/precios) a un
    PriceRequest. Nombres de columna canónicos con defaults tolerantes; documentar el
    mapeo esperado en el README del caso."""
    competitor_avg_price = _num(record, "competitor_avg_price", None)

    return PriceRequest(
        product_id=str(record.get("product_id", "unknown")),
        base_cost=_num(record, "base_cost", 1.0),
        user_segment=str(record.get("user_segment", "default") or "default"),
        inventory_level_pct=_num(record, "inventory_level_pct", 50.0),
        competitor_avg_price=competitor_avg_price,
        demand_index=_num(record, "demand_index", 0.5),
        min_margin_pct=_num(record, "min_margin_pct", 0.10),
        max_multiplier=_num(record, "max_multiplier", 2.5),
        min_multiplier=_num(record, "min_multiplier", 1.0),
    )


def row_to_vector(row: pd.Series) -> np.ndarray:
    """Atajo usado por el entorno de entrenamiento: fila de DataFrame -> vector de estado."""
    return request_to_vector(record_to_request(row.to_dict()))
