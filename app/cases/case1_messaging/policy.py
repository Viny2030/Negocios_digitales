"""
Wrapper de la política entrenada (Caso 1.1). Carga un modelo stable-baselines3
guardado por empresa (storage/models/<company_id>/case1_messaging/model.zip).
Si no existe modelo entrenado todavía, usa una heurística de fallback para que
el endpoint funcione desde el día 1 (útil para demo y para integraciones tempranas).
"""

import os

from app.cases.case1_messaging.encoding import profile_to_vector
from app.cases.case1_messaging.schemas import ProspectProfile, Touchpoint

try:
    from stable_baselines3 import DQN, PPO
except ImportError:
    DQN = PPO = None


class MessagingPolicy:
    def __init__(self, model_path: str | None = None):
        self.model = None
        if model_path and os.path.exists(model_path) and DQN is not None:
            loader = PPO if "ppo" in model_path.lower() else DQN
            self.model = loader.load(model_path)

    def _heuristic(self, profile: ProspectProfile) -> tuple[Touchpoint, float, str]:
        """Regla de respaldo, simple pero razonable, mientras no hay modelo RL entrenado."""
        if profile.unsubscribed:
            return Touchpoint.wait, 0.0, "El prospecto se dio de baja; no contactar."
        if profile.days_since_last_contact < 1:
            return Touchpoint.wait, 0.1, "Contacto reciente; esperar para evitar fatiga."
        if profile.purchase_stage == "decision":
            return Touchpoint.call, 0.7, "Etapa de decisión: llamada tiene mayor tasa de cierre."
        if "email_a" not in profile.channels_contacted:
            return Touchpoint.email_a, 0.5, "Primer touchpoint recomendado: email variante A."
        return Touchpoint.retargeting_meta, 0.4, "Reforzar con retargeting antes de escalar canal."

    def predict(self, profile: ProspectProfile) -> tuple[Touchpoint, float, str]:
        if self.model is None:
            return self._heuristic(profile)

        obs = profile_to_vector(profile)
        action_idx, _ = self.model.predict(obs, deterministic=True)
        action = list(Touchpoint)[int(action_idx)]
        # value_estimate real requeriría acceder a la red de valor del modelo;
        # se deja un placeholder para mantener el contrato de la respuesta.
        return action, 0.5, "Recomendación generada por el modelo RL entrenado."
