"""
Evaluación de equidad (Demographic Parity) del Caso 2.1, vía Fairlearn.

Se corre offline, después de entrenar (ver scripts/train_case2.py), sobre el propio
historical_df usado para entrenamiento. Simula qué multiplicador habría asignado la
política ya entrenada a cada fila, y define como "resultado favorable" recibir un
multiplicador por debajo de la mediana global (es decir, un precio relativamente más
bajo dentro del rango de la empresa). demographic_parity_difference mide la mayor
diferencia en la tasa de ese resultado favorable entre segmentos (`user_segment`):
un valor alto indica que el agente favorece sistemáticamente a un segmento sobre otro.

Requiere la columna `user_segment` en el dataset histórico; si no está presente, se
omite la evaluación (no se bloquea el despliegue por falta de datos, pero queda
constancia explícita de que no se pudo evaluar).
"""

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, demographic_parity_difference, selection_rate

from app.cases.case2_pricing.encoding import row_to_vector


def evaluate_fairness(model, historical_df: pd.DataFrame, min_multiplier: float, max_multiplier: float) -> dict:
    if model is None or historical_df is None or len(historical_df) == 0:
        return {
            "evaluated": False,
            "reason": "sin modelo o dataset histórico vacío",
            "demographic_parity_difference": None,
        }

    if "user_segment" not in historical_df.columns:
        return {
            "evaluated": False,
            "reason": "el dataset no tiene columna 'user_segment'",
            "demographic_parity_difference": None,
        }

    multipliers = np.array(
        [
            float(
                np.clip(
                    model.predict(row_to_vector(row), deterministic=True)[0][0], min_multiplier, max_multiplier
                )
            )
            for _, row in historical_df.iterrows()
        ]
    )
    favorable = (multipliers <= np.median(multipliers)).astype(int)  # 1 = precio relativamente bajo
    sensitive_features = historical_df["user_segment"].astype(str).values

    dp_diff = float(
        demographic_parity_difference(
            y_true=favorable,
            y_pred=favorable,
            sensitive_features=sensitive_features,
        )
    )
    frame = MetricFrame(
        metrics=selection_rate, y_true=favorable, y_pred=favorable, sensitive_features=sensitive_features
    )

    return {
        "evaluated": True,
        "demographic_parity_difference": round(dp_diff, 4),
        "selection_rate_by_segment": {str(k): round(float(v), 4) for k, v in frame.by_group.items()},
        "n_samples": int(len(historical_df)),
    }
