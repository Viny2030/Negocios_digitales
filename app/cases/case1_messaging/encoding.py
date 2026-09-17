"""
Codificación de estado del Caso 1.1 — única fuente de verdad, usada tanto por
`policy.py` (inferencia) como por `env.py` (entrenamiento offline con historical_df).
Antes vivían dos copias del mismo mapeo state->vector: una en `MessagingPolicy` y otra
implícita (y nunca completada) en `MessagingSequencingEnv._sample_prospect`. Tenerlas
separadas es peligroso: si difieren, el modelo se entrena con una codificación y se
consulta con otra, y las predicciones dejan de tener sentido.
"""

import numpy as np
import pandas as pd

from app.cases.case1_messaging.schemas import ProspectProfile

CHANNELS = ["email_a", "email_b", "email_c", "retargeting_meta", "sms", "push", "call"]
PURCHASE_STAGES = ["awareness", "consideration", "decision"]

STATE_DIM = len(CHANNELS) + 3 + len(PURCHASE_STAGES) + 1  # 7 + 3 + 3 + 1 = 14


def profile_to_vector(profile: ProspectProfile) -> np.ndarray:
    """Vector de estado (S): [channels_contacted_onehot(7), messages_received,
    responses_count, days_since_last_contact, purchase_stage_onehot(3), unsubscribed]."""
    onehot_channels = [1.0 if c in profile.channels_contacted else 0.0 for c in CHANNELS]
    onehot_stage = [1.0 if profile.purchase_stage == s else 0.0 for s in PURCHASE_STAGES]
    vector = (
        onehot_channels
        + [
            min(profile.messages_received / 10.0, 1.0),
            min(profile.responses_count / 5.0, 1.0),
            min(profile.days_since_last_contact / 30.0, 1.0),
        ]
        + onehot_stage
        + [1.0 if profile.unsubscribed else 0.0]
    )
    return np.array(vector, dtype=np.float32)


def _num(row, key: str, default: float) -> float:
    """Lee un campo numérico de una fila/registro tolerando NaN/None -- `row.get(key,
    default) or default` NO alcanza acá porque un NaN de pandas es "truthy" en Python
    (no es falsy como 0 o None), así que se colaba tal cual en vez de caer al default."""
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


def _bool(row, key: str, default: bool) -> bool:
    """Igual que `_num` pero para flags booleanos (ej. `unsubscribed`): un NaN no debe
    interpretarse como True."""
    value = row.get(key, default)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return bool(value)


def _parse_channels_contacted(raw) -> list[str]:
    """El CSV del cliente no puede tener listas nativas: `channels_contacted` viaja
    como string separado por coma o punto y coma (ej. "email_a;sms"). Si ya viene
    como lista (por ejemplo, un Parquet o un test), se respeta tal cual."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return raw
    text = str(raw).strip()
    if not text:
        return []
    separator = ";" if ";" in text else ","
    return [c.strip() for c in text.split(separator) if c.strip()]


def record_to_profile(record: dict) -> ProspectProfile:
    """Mapea una fila del dataset histórico de la empresa (exportado de HubSpot/
    Salesforce/etc., ver app/api/v1/endpoints/datasets.py) a un ProspectProfile.
    Usa nombres de columna canónicos con defaults tolerantes, porque cada CRM
    exporta con sus propios nombres; documentar el mapeo esperado en el README
    del caso para que la empresa ajuste su export si hace falta."""
    raw_stage = record.get("purchase_stage", "awareness")
    if raw_stage is None or (isinstance(raw_stage, float) and pd.isna(raw_stage)):
        raw_stage = "awareness"
    purchase_stage = str(raw_stage).strip() or "awareness"
    return ProspectProfile(
        prospect_id=str(record.get("prospect_id", "unknown")),
        channels_contacted=_parse_channels_contacted(record.get("channels_contacted")),
        messages_received=int(_num(record, "messages_received", 0.0)),
        responses_count=int(_num(record, "responses_count", 0.0)),
        days_since_last_contact=_num(record, "days_since_last_contact", 0.0),
        purchase_stage=str(purchase_stage),
        unsubscribed=_bool(record, "unsubscribed", False),
    )


def row_to_vector(row: pd.Series) -> np.ndarray:
    """Atajo usado por el entorno de entrenamiento: fila de DataFrame -> vector de estado."""
    return profile_to_vector(record_to_profile(row.to_dict()))
