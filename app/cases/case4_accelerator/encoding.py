"""
Codificación de estado del Caso 6.2 — única fuente de verdad, usada tanto por
`policy.py` (inferencia) como por `env.py` (entrenamiento offline con historical_df).
Ver app/cases/case1_messaging/encoding.py para la misma razón de ser.

A diferencia de los otros 3 casos, acá el "estado" no es un solo prospecto/precio/
alerta sino un portfolio completo de startups. Como stable-baselines3 necesita un
tamaño de observación y de acción FIJO, el portfolio se vectoriza con padding/truncado
a MAX_STARTUPS slots (startups de más se procesan en tandas desde policy.py).
"""

import numpy as np
import pandas as pd

from app.cases.case4_accelerator.schemas import ResourcePool, StartupProfile

MAX_STARTUPS = 15
STARTUP_FEATURES = 7  # traction, team, market, runway_norm, past_response, mentorship_recv_norm, capital_recv_norm
SLOT_FEATURES = STARTUP_FEATURES + 1  # +1: flag "slot activo" (para distinguir startup real de padding)
GLOBAL_FEATURES = 3  # mentoria_disponible_norm, capital_disponible_norm, conexiones_disponibles_norm
STATE_DIM = MAX_STARTUPS * SLOT_FEATURES + GLOBAL_FEATURES  # 15*8 + 3 = 123


def _global_features(resources: ResourcePool) -> list[float]:
    return [
        min(resources.mentorship_hours_available / 200.0, 1.0),
        min(resources.followon_capital_available / 1_000_000.0, 1.0),
        min(resources.strategic_connections_available / 20.0, 1.0),
    ]


def _startup_slot_vector(startup: StartupProfile | None) -> list[float]:
    """Vector de un slot del portfolio: startup real (flag activo=1) o slot vacío
    (todo ceros, flag activo=0) para completar hasta MAX_STARTUPS."""
    if startup is None:
        return [0.0] * SLOT_FEATURES
    return [
        1.0,  # slot activo
        startup.traction_score,
        startup.team_score,
        startup.market_score,
        min(startup.runway_months / 24.0, 1.0),
        startup.past_response_score,
        min(startup.mentorship_hours_received_last_period / 40.0, 1.0),
        min(startup.followon_capital_received_last_period / 100_000.0, 1.0),
    ]


def portfolio_to_vector(startups: list[StartupProfile], resources: ResourcePool) -> np.ndarray:
    """Observación (S) completa de una tanda de portfolio (hasta MAX_STARTUPS
    startups), con padding si hay menos. Si el portfolio real tiene más de
    MAX_STARTUPS, quien llama (policy.py) es responsable de partirlo en tandas."""
    slots = startups[:MAX_STARTUPS]
    vector: list[float] = []
    for i in range(MAX_STARTUPS):
        startup = slots[i] if i < len(slots) else None
        vector.extend(_startup_slot_vector(startup))
    vector.extend(_global_features(resources))
    return np.array(vector, dtype=np.float32)


def record_to_startup(record: dict) -> StartupProfile:
    """Mapea una fila del dataset histórico de portfolios (Crunchbase / YC-500
    Startups open data / Endeavor LatAm / sintético) a un StartupProfile. Nombres de
    columna canónicos con defaults tolerantes; documentar el mapeo esperado en el
    README del caso."""
    return StartupProfile(
        startup_id=str(record.get("startup_id", "unknown")),
        traction_score=float(record.get("traction_score", 0.5) or 0.5),
        team_score=float(record.get("team_score", 0.5) or 0.5),
        market_score=float(record.get("market_score", 0.5) or 0.5),
        runway_months=float(record.get("runway_months", 6.0) or 6.0),
        past_response_score=float(record.get("past_response_score", 0.5) or 0.5),
        mentorship_hours_received_last_period=float(
            record.get("mentorship_hours_received_last_period", 0.0) or 0.0
        ),
        followon_capital_received_last_period=float(
            record.get("followon_capital_received_last_period", 0.0) or 0.0
        ),
    )


def sample_portfolio_from_df(df: pd.DataFrame, n: int = MAX_STARTUPS) -> list[StartupProfile]:
    """Muestrea un portfolio de hasta `n` startups reales del dataset histórico de la
    empresa, para calibrar el entorno de entrenamiento con datos reales (mismo
    principio que encoding.py en los casos 1-3: nada de descartar la fila muestreada)."""
    sampled = df.sample(min(n, len(df)))
    return [record_to_startup(row.to_dict()) for _, row in sampled.iterrows()]
