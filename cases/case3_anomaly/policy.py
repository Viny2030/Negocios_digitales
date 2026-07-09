"""Wrapper de la política RL de triaje (DQN sobre cola priorizada), con fallback heurístico."""
import os

import numpy as np

from app.cases.case3_anomaly.schemas import AlertWebhookPayload, TeamCapacity, TriageAction

try:
    from stable_baselines3 import DQN
except ImportError:
    DQN = None


class AnomalyTriagePolicy:
    def __init__(self, model_path: str | None = None):
        self.model = None
        if model_path and os.path.exists(model_path) and DQN is not None:
            self.model = DQN.load(model_path)

    @staticmethod
    def _to_vector(payload: AlertWebhookPayload, capacity: TeamCapacity, fp_proba: float) -> np.ndarray:
        return np.array([
            min(abs(payload.z_score) / 5.0, 1.0),
            0.0,  # tiempo transcurrido (se puede derivar de triggered_at vs. now)
            min(payload.historical_similar_count / 20.0, 1.0),
            min(capacity.analyst_hours_available / 8.0, 1.0),
            min(capacity.pending_alerts_in_queue / 50.0, 1.0),
            min((payload.estimated_financial_impact or 0) / 10000.0, 1.0),
        ], dtype=np.float32)

    def _heuristic(self, payload: AlertWebhookPayload, capacity: TeamCapacity, fp_proba: float) -> TriageAction:
        if fp_proba > 0.7:
            return TriageAction.ignore
        if capacity.analyst_hours_available < 1:
            return TriageAction.escalate_engineering  # sin capacidad, delegar
        if abs(payload.z_score) > 3 and (payload.estimated_financial_impact or 0) > 5000:
            return TriageAction.investigate_deep
        if abs(payload.z_score) > 2:
            return TriageAction.investigate_light
        return TriageAction.merge_related if payload.historical_similar_count > 3 else TriageAction.investigate_light

    def decide(
        self, payload: AlertWebhookPayload, capacity: TeamCapacity, fp_proba: float,
    ) -> tuple[TriageAction, float]:
        if self.model is None:
            action = self._heuristic(payload, capacity, fp_proba)
            priority_score = min(abs(payload.z_score) / 5.0, 1.0) * (1 - fp_proba)
            return action, round(priority_score, 3)

        obs = self._to_vector(payload, capacity, fp_proba)
        action_idx, _ = self.model.predict(obs, deterministic=True)
        action = list(TriageAction)[int(action_idx)]
        priority_score = min(abs(payload.z_score) / 5.0, 1.0) * (1 - fp_proba)
        return action, round(priority_score, 3)
