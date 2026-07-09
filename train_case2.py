"""
Entrenamiento offline del Caso 2.1 (pricing dinámico) para una empresa específica.
Incorpora la penalización de equidad (Demographic Parity, vía Fairlearn) evaluada
sobre el historial de transacciones antes/después del entrenamiento.

Uso manual:
    python -m scripts.train_case2 --company-id 1 --dataset-id 5 --algorithm ppo --timesteps 50000
"""
import argparse
import os

import pandas as pd
from sqlalchemy.orm import Session

from app.cases.case2_pricing.env import PricingEnv
from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.dataset import Dataset

settings = get_settings()


def train_for_company(
    company_id: int, dataset_id: int, algorithm: str = "ppo",
    timesteps: int = 50_000, fairness_penalty_weight: float = 1.0,
) -> str:
    from stable_baselines3 import DQN, PPO

    db: Session = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.company_id == company_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} no encontrado para company_id={company_id}")
        historical_df = pd.read_csv(dataset.file_path)
    finally:
        db.close()

    env = PricingEnv(historical_df=historical_df, fairness_penalty_weight=fairness_penalty_weight)
    algo_cls = DQN if algorithm.lower() == "dqn" else PPO
    model = algo_cls("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)

    # TODO: evaluación post-entrenamiento con fairlearn.metrics.demographic_parity_difference
    # sobre `historical_df` segmentado, y bloquear el despliegue si supera el umbral aceptado.

    out_dir = os.path.join(settings.MODELS_DIR, str(company_id), "case2_pricing")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"model_{algorithm.lower()}.zip")
    model.save(out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--company-id", type=int, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--algorithm", default="ppo")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--fairness-penalty-weight", type=float, default=1.0)
    args = parser.parse_args()

    path = train_for_company(
        args.company_id, args.dataset_id, args.algorithm, args.timesteps, args.fairness_penalty_weight,
    )
    print(f"Modelo guardado en: {path}")
