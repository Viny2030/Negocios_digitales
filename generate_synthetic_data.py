"""Genera los CSV sintéticos de synthetic_data/ para desarrollo/tests.

AGREGADO 2026-09-22: reescrito para el pivot a DynamicPricing/SmartRoute/
LeadMatch. De los 3, solo LeadMatch entrena algo con un dataset subido (ver
CaseType en app/models/dataset.py) -- por eso este script genera un único CSV
en vez de los 4 de antes."""

import os

import numpy as np

random_state = np.random.default_rng(42)
OUT = "synthetic_data"
os.makedirs(OUT, exist_ok=True)

import pandas as pd  # noqa: E402 (después de fijar la semilla, por claridad)

n = 600
ORIGINS = ["organic", "paid_ads", "referral", "event", "outbound"]
origin = random_state.choice(ORIGINS, n, p=[0.3, 0.3, 0.15, 0.1, 0.15])
time_on_site_minutes = np.round(random_state.exponential(scale=12, size=n), 1)
company_size = random_state.integers(1, 2000, n)
rep_id = random_state.choice([f"rep-{i}" for i in range(1, 9)], n)

# La probabilidad "real" de conversión combina tiempo en sitio, tamaño de empresa y
# origen (paid_ads y referral convierten mejor que outbound en este dataset sintético),
# más ruido -- para que el clasificador entrenado tenga señal real que aprender en vez
# de una etiqueta puramente aleatoria (ver tests/test_lead_match.py, que verifica que
# el modelo entrenado con esto queda por encima de la heurística de respaldo).
origin_bias = (
    pd.Series(origin)
    .map({"organic": 0.0, "paid_ads": 0.4, "referral": 0.5, "event": 0.1, "outbound": -0.3})
    .to_numpy()
)
score = 0.04 * time_on_site_minutes + 0.0015 * company_size + origin_bias + random_state.normal(0, 1.0, n)
converted = (score > np.median(score)).astype(int)

pd.DataFrame(
    {
        "lead_id": [f"lead-{i}" for i in range(n)],
        "origin": origin,
        "time_on_site_minutes": time_on_site_minutes,
        "company_size": company_size,
        "rep_id": rep_id,
        "converted": converted,
    }
).to_csv(f"{OUT}/lead_match.csv", index=False)

print("CSV generado en", OUT)
