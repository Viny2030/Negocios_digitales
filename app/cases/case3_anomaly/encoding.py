"""
Codificación de estado del Caso 3.3 — única fuente de verdad, usada tanto por
`policy.py` (inferencia) como por `env.py` (entrenamiento offline con historical_df).
Ver app/cases/case1_messaging/encoding.py para la misma razón de ser.

Nota: la probabilidad de falso positivo (fp_proba) NO forma parte del vector de
estado — el vector original de policy.py recibía ese parámetro pero nunca lo usaba
(bug de la versión anterior). Se deja fuera explícitamente en vez de agregarlo
"para que cierre", porque agregarlo ahora cambiaría la forma del vector de un
modelo ya entrenado con la versión vieja; si se quiere incorporar, es un cambio de
`STATE_DIM` que requiere reentrenar desde cero, no un parche silencioso.
"""
import datetime as dt

import numpy as np
import pandas as pd

from app.cases.case3_anomaly.schemas import AlertWebhookPayload, TeamCapacity

STATE_DIM = 6


def payload_to_vector(payload: AlertWebhookPayload, capacity: TeamCapacity) -> np.ndarray:
    """Vector de estado (S): [z_score_norm, tiempo_transcurrido_norm(placeholder),
    frecuencia_mensual_norm, horas_analista_disponibles_norm, alertas_pendientes_norm,
    impacto_estimado_norm]."""
    return np.array([
        min(abs(payload.z_score) / 5.0, 1.0),
        0.0,  # tiempo transcurrido: se puede derivar de triggered_at vs. now (ver TODO abajo)
        min(payload.historical_similar_count / 20.0, 1.0),
        min(capacity.analyst_hours_available / 8.0, 1.0),
        min(capacity.pending_alerts_in_queue / 50.0, 1.0),
        min((payload.estimated_financial_impact or 0) / 10000.0, 1.0),
    ], dtype=np.float32)


def record_to_payload(record: dict) -> AlertWebhookPayload:
    """Mapea una fila del dataset histórico (NAB / Yahoo Anomaly Dataset / tickets
    propios) a un AlertWebhookPayload. Nombres de columna canónicos con defaults
    tolerantes; documentar el mapeo esperado en el README del caso."""
    pct_change = record.get("pct_change")
    if pct_change is not None and isinstance(pct_change, float) and pd.isna(pct_change):
        pct_change = None

    estimated_financial_impact = record.get("estimated_financial_impact")
    if estimated_financial_impact is not None and isinstance(estimated_financial_impact, float) \
            and pd.isna(estimated_financial_impact):
        estimated_financial_impact = None

    return AlertWebhookPayload(
        alert_id=str(record.get("alert_id", "unknown")),
        metric_name=str(record.get("metric_name", "unknown")),
        z_score=float(record.get("z_score", 0.0) or 0.0),
        pct_change=pct_change,
        triggered_at=record.get("triggered_at") or dt.datetime.utcnow(),
        business_context=record.get("business_context"),
        historical_similar_count=int(record.get("historical_similar_count", 0) or 0),
        estimated_financial_impact=estimated_financial_impact,
    )


def sample_capacity() -> TeamCapacity:
    """El dataset histórico de alertas no trae la capacidad del equipo en ese momento
    (es estado operativo, no un atributo de la alerta), así que se simula dentro de
    rangos realistas para el entrenamiento offline."""
    return TeamCapacity(
        analyst_hours_available=float(np.random.uniform(0, 8)),
        pending_alerts_in_queue=int(np.random.randint(0, 50)),
    )


def row_to_vector(row: pd.Series) -> np.ndarray:
    """Atajo usado por el entorno de entrenamiento: fila de DataFrame -> vector de estado."""
    payload = record_to_payload(row.to_dict())
    capacity = sample_capacity()
    return payload_to_vector(payload, capacity)