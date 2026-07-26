"""Wrapper de la política de pricing entrenada (PPO/DQN) por empresa, con fallback heurístico."""
import os

import numpy as np

from app.cases.case2_pricing.encoding import request_to_vector
from app.cases.case2_pricing.schemas import PriceRequest

try:
    from stable_baselines3 import DQN, PPO
except ImportError:
    DQN = PPO = None


class PricingPolicy:
    def __init__(self, model_path: str | None = None):
        self.model = None
        if model_path and os.path.exists(model_path) and PPO is not None:
            loader = DQN if "dqn" in model_path.lower() else PPO
            self.model = loader.load(model_path)

    def _heuristic(self, request: PriceRequest) -> float:
        """Precio de respaldo: ajusta el multiplicador según inventario y demanda,
        antes de tener un modelo RL entrenado con datos reales de la empresa."""
        scarcity_factor = 1.0 - (request.inventory_level_pct / 100.0)  # menos stock -> precio más alto
        multiplier = request.min_multiplier + (request.max_multiplier - request.min_multiplier) * (
            0.5 * scarcity_factor + 0.5 * request.demand_index
        )
        return request.base_cost * multiplier

    def predict_price(self, request: PriceRequest) -> float:
        if self.model is None:
            return self._heuristic(request)

        obs = request_to_vector(request)
        action, _ = self.model.predict(obs, deterministic=True)
        multiplier = float(np.clip(action[0], request.min_multiplier, request.max_multiplier))
        return request.base_cost * multiplier