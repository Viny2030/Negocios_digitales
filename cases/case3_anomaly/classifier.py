"""
Clasificador auxiliar (Caso 3.3): estima la probabilidad de que una alerta sea un falso
positivo (ruido) vs. una anomalía real, ANTES de que el agente RL decida el curso de acción.
Se entrena con logs históricos de tickets resueltos (columna label: real_incident/false_positive).
"""
import os

import joblib
import numpy as np

from app.cases.case3_anomaly.schemas import AlertWebhookPayload


class FalsePositiveClassifier:
    def __init__(self, model_path: str | None = None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)

    @staticmethod
    def _payload_to_features(payload: AlertWebhookPayload) -> np.ndarray:
        return np.array([[
            abs(payload.z_score),
            payload.pct_change or 0.0,
            payload.historical_similar_count,
            1.0 if payload.estimated_financial_impact else 0.0,
        ]])

    def predict_false_positive_proba(self, payload: AlertWebhookPayload) -> float:
        if self.model is None:
            # Heurística: mayor historial de alertas similares repetidas -> más probable que sea ruido conocido
            baseline = 0.5 - min(abs(payload.z_score) / 10.0, 0.4)
            repeat_penalty = min(payload.historical_similar_count * 0.03, 0.3)
            return float(np.clip(baseline + repeat_penalty, 0.0, 1.0))

        features = self._payload_to_features(payload)
        proba = self.model.predict_proba(features)[0]
        # Se asume orden de clases [real_incident, false_positive]; ajustar según el entrenamiento real
        return float(proba[1])
