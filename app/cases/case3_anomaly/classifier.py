"""
Clasificador auxiliar (Caso 3.3): estima la probabilidad de que una alerta sea un falso
positivo (ruido) vs. una anomalía real, ANTES de que el agente RL decida el curso de acción.
Se entrena con logs históricos de tickets resueltos (columna `is_false_positive`:
1 = false_positive, 0 = real_incident; se acepta `label` como alias legado -- ver
scripts/train_case3.py._train_fp_classifier y app/cases/case3_anomaly/env.py, que leen
la misma columna con el mismo fallback).
"""

import os

import joblib
import numpy as np

from app.cases.case3_anomaly.schemas import AlertWebhookPayload

# Debe coincidir exactamente (nombres y orden) con `feature_cols` en
# scripts/train_case3.py::_train_fp_classifier. Se usa como fallback cuando la versión
# promovida no trae `classifier_features` en sus metrics (modelos entrenados antes de
# que se empezara a persistir esa lista, o si el dataset de entrenamiento traía todas
# las columnas).
DEFAULT_FEATURE_COLS = ["z_score", "pct_change", "historical_similar_count", "estimated_financial_impact"]


class FalsePositiveClassifier:
    def __init__(self, model_path: str | None = None, feature_cols: list[str] | None = None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
        # Columnas (y orden) con las que se entrenó realmente el modelo cargado -- ver
        # `available_cols` en scripts/train_case3.py. Si el dataset de entrenamiento no traía
        # alguna columna, el modelo se entrenó con menos features; hay que reconstruir el vector
        # de inferencia con exactamente esas mismas columnas para no romper
        # `model.predict_proba` (ValueError: X has N features, but ... expects M) ni volver a
        # introducir el desajuste train/serve.
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS

    def _payload_to_features(self, payload: AlertWebhookPayload) -> np.ndarray:
        # OJO: estos valores deben ser exactamente los mismos (mismo signo, misma magnitud, sin
        # binarizar) que las columnas homónimas que usa `_train_fp_classifier` en
        # scripts/train_case3.py al entrenar, o el modelo recibe en inferencia features que no
        # tienen nada que ver con lo que aprendió.
        available_values = {
            "z_score": payload.z_score,
            "pct_change": payload.pct_change or 0.0,
            "historical_similar_count": payload.historical_similar_count,
            "estimated_financial_impact": payload.estimated_financial_impact or 0.0,
        }
        row = [available_values[col] for col in self.feature_cols]
        return np.array([row])

    def predict_false_positive_proba(self, payload: AlertWebhookPayload) -> float:
        if self.model is None:
            # Heurística: mayor historial de alertas similares repetidas -> más probable que sea ruido conocido
            baseline = 0.5 - min(abs(payload.z_score) / 10.0, 0.4)
            repeat_penalty = min(payload.historical_similar_count * 0.03, 0.3)
            return float(np.clip(baseline + repeat_penalty, 0.0, 1.0))

        features = self._payload_to_features(payload)
        proba = self.model.predict_proba(features)[0]
        # No asumir que `classes_` siempre tiene longitud 2 y que el índice 1 es la clase
        # "false_positive": si el historial de la empresa todavía no tiene tickets de ambas
        # clases (dataset nuevo con `is_false_positive` todo 0 o todo 1, algo realista para una
        # empresa con pocos tickets), `model.classes_` tiene un solo elemento y `proba[1]`
        # explota con IndexError. Mapear por la etiqueta real de clase evita eso: si sólo existe
        # la clase 0, la probabilidad de clase 1 es 0.0; si sólo existe la clase 1, todo es esa
        # clase y su probabilidad es el único valor de `proba`.
        proba_by_class = dict(zip(self.model.classes_, proba, strict=True))
        return float(proba_by_class.get(1, 0.0))
