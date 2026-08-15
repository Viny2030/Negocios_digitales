"""
Wrapper de la política de asignación (Caso 6.2). Estima, por startup del portfolio,
un score de valor esperado por unidad de recurso dado su estado y el contexto de
escasez de recursos del período. Carga un modelo stable-baselines3 (PPO, ver env.py:
el espacio de acciones es continuo) entrenado sobre el portfolio completo si existe;
si no, cae a una heurística de reglas simples ponderando tracción, equipo, mercado y
respuesta a intervenciones previas — útil para que el endpoint funcione desde el día
1, igual que en los otros 3 casos.
"""

import os

import numpy as np

from app.cases.case4_accelerator.encoding import MAX_STARTUPS, portfolio_to_vector
from app.cases.case4_accelerator.schemas import ResourcePool, StartupProfile

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None


class AcceleratorPolicy:
    def __init__(self, model_path: str | None = None):
        self.model = None
        if model_path and os.path.exists(model_path) and PPO is not None:
            self.model = PPO.load(model_path)

    @staticmethod
    def _heuristic_score(startup: StartupProfile) -> float:
        """Score de respaldo: pondera tracción, equipo y mercado (fundamentals) y
        premia buena respuesta histórica a intervenciones (la inversión rinde más en
        startups que ya demostraron aprovechar el apoyo recibido)."""
        fundamentals = 0.4 * startup.traction_score + 0.3 * startup.team_score + 0.3 * startup.market_score
        responsiveness_multiplier = 0.6 + 0.4 * startup.past_response_score
        return float(np.clip(fundamentals * responsiveness_multiplier, 0.0, 1.0))

    def score_portfolio(self, startups: list[StartupProfile], resources: ResourcePool) -> dict[str, float]:
        """Devuelve {startup_id: score_de_valor_esperado}, insumo del LP de
        constraints.py. Si el portfolio real tiene más de MAX_STARTUPS startups, se
        procesa en tandas porque el modelo RL tiene un tamaño de observación/acción
        fijo (ver encoding.py); cada tanda es independiente entre sí, así que el
        score de una startup no depende de en qué tanda cayó — es una limitación
        conocida del MVP, no una decisión de diseño ideal."""
        if self.model is None:
            return {s.startup_id: self._heuristic_score(s) for s in startups}

        scores: dict[str, float] = {}
        for i in range(0, len(startups), MAX_STARTUPS):
            batch = startups[i : i + MAX_STARTUPS]
            obs = portfolio_to_vector(batch, resources)
            raw_action, _ = self.model.predict(obs, deterministic=True)
            for startup, raw_score in zip(batch, raw_action[: len(batch)]):
                scores[startup.startup_id] = float(np.clip(raw_score, 0.0, 1.0))
        return scores
