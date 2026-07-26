"""
Codificación de estado del Caso 2.1 — única fuente de verdad, usada tanto por
`policy.py` (inferencia) como por `env.py` (entrenamiento offline con historical_df).
Ver app/cases/case1_messaging/encoding.py para la misma razón de ser: entrenar y
predecir con codificaciones distintas invalida el modelo.
"""
import numpy as np
import pandas as pd

from app.cases.case2_pricing.schemas import PriceRequest

STATE_DIM = 6


def request_to_vector(request: PriceRequest) -> np.ndarray:
    """Vector de estado (S): [inventory_level_pct_norm, base_cost_norm, segment_encoded,
    rejection_history(placeholder), competitor_price_norm, demand_index]."""
    competitor_norm = 0.5
    if request.competitor_avg_price:
        competitor_norm = min(request.competitor_avg_price / (request.base_cost * request.max_multiplier), 1.0)
    vector = [
        request.inventory_level_pct / 100.0,
        min(request.base_cost / 1000.0, 1.0),
        hash(request.user_segment) % 100 / 100.0,  # placeholder de encoding de segmento
        0.0,  # historial de rechazo (vendría del dataset/CRM)
        competitor_norm,
        request.demand_index,
    ]
    return np.array(vector, dtype=np.float32)


def record_to_request(record: dict) -> PriceRequest:
    """Mapea una fila del dataset histórico de la empresa (transacciones/precios) a un
    PriceRequest. Nombres de columna canónicos con defaults tolerantes; documentar el
    mapeo esperado en el README del caso."""
    competitor_avg_price = record.get("competitor_avg_price")
    if competitor_avg_price is not None and isinstance(competitor_avg_price, float) and pd.isna(competitor_avg_price):
        competitor_avg_price = None

    return PriceRequest(
        product_id=str(record.get("product_id", "unknown")),
        base_cost=float(record.get("base_cost", 1.0) or 1.0),
        user_segment=str(record.get("user_segment", "default") or "default"),
        inventory_level_pct=float(record.get("inventory_level_pct", 50.0) or 50.0),
        competitor_avg_price=competitor_avg_price,
        demand_index=float(record.get("demand_index", 0.5) or 0.5),
        min_margin_pct=float(record.get("min_margin_pct", 0.10) or 0.10),
        max_multiplier=float(record.get("max_multiplier", 2.5) or 2.5),
        min_multiplier=float(record.get("min_multiplier", 1.0) or 1.0),
    )


def row_to_vector(row: pd.Series) -> np.ndarray:
    """Atajo usado por el entorno de entrenamiento: fila de DataFrame -> vector de estado."""
    return request_to_vector(record_to_request(row.to_dict()))