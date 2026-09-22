"""
Scoring predictivo de conversión: regresión logística sobre las features del
lead (origen, tiempo en sitio, tamaño de empresa), entrenada por empresa con
sus propios leads históricos (ver train.py). Si todavía no hay modelo
entrenado para la empresa, cae a una heurística de respaldo -- mismo patrón
que el resto de la plataforma (ver policy._heuristic de los casos anteriores),
acá aplicado a un clasificador simple en vez de una política RL.
"""

import math
from dataclasses import dataclass

import joblib

from app.cases.lead_match.schemas import LeadFeatures

# Pesos de la heurística de respaldo: mayor tiempo en sitio y empresas más
# grandes se asocian a mayor intención/potencial de compra. Los denominadores
# son de escala (30 minutos y 500 empleados "saturan" el score), no vienen de
# un ajuste estadístico -- es un prior razonable para no devolver 0.5 fijo
# antes de que la empresa tenga datos propios para entrenar.
_HEURISTIC_TIME_SCALE = 30.0
_HEURISTIC_SIZE_SCALE = 500.0


@dataclass
class ConversionScorer:
    model: object | None  # sklearn Pipeline entrenado, o None si no hay modelo promovido
    is_default: bool

    def predict_proba(self, lead: LeadFeatures) -> float:
        if self.model is None:
            return self._heuristic_score(lead)
        import pandas as pd

        row = pd.DataFrame(
            [
                {
                    "origin": lead.origin,
                    "time_on_site_minutes": lead.time_on_site_minutes,
                    "company_size": lead.company_size,
                }
            ]
        )
        proba = self.model.predict_proba(row)[0]
        # predict_proba devuelve [P(clase=0), P(clase=1)] -- converted=1 es la positiva.
        classes = list(self.model.classes_)
        return float(proba[classes.index(1)]) if 1 in classes else float(proba[-1])

    @staticmethod
    def _heuristic_score(lead: LeadFeatures) -> float:
        time_component = min(lead.time_on_site_minutes / _HEURISTIC_TIME_SCALE, 1.0)
        size_component = min(lead.company_size / _HEURISTIC_SIZE_SCALE, 1.0)
        # Combina ambos componentes y los pasa por una sigmoide centrada en 0.5
        # para que el resultado se comporte como una probabilidad (0-1, con la
        # mayor densidad cerca del medio en vez de pegada a los extremos).
        raw = (time_component + size_component) / 2
        return 1 / (1 + math.exp(-6 * (raw - 0.5)))


def load_scorer(model_path: str | None) -> ConversionScorer:
    if model_path is None:
        return ConversionScorer(model=None, is_default=True)
    model = joblib.load(model_path)
    return ConversionScorer(model=model, is_default=False)
