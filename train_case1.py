"""
Entrenamiento offline del Caso 1.1 (secuenciación de mensajes) para una empresa específica.

Uso manual:
    python -m scripts.train_case1 --company-id 1 --dataset-id 3 --algorithm dqn --timesteps 50000

También es invocado en background desde app/cases/case1_messaging/router.py (endpoint /train).
"""
import argparse
import os

import pandas as pd
from sqlalchemy.orm import Session

from app.cases.case1_messaging.env import MessagingSequencingEnv
from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.dataset import Dataset

settings = get_settings()


def train_for_company(company_id: int, dataset_id: int, algorithm: str = "dqn", timesteps: int = 50_000) -> str:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.env_util import make_vec_env

    db: Session = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.company_id == company_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} no encontrado para company_id={company_id}")
        historical_df = pd.read_csv(dataset.file_path)
    finally:
        db.close()

    env = MessagingSequencingEnv(historical_df=historical_df)
    algo_cls = PPO if algorithm.lower() == "ppo" else DQN
    model = algo_cls("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)

    out_dir = os.path.join(settings.MODELS_DIR, str(company_id), "case1_messaging")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"model_{algorithm.lower()}.zip")
    model.save(out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--company-id", type=int, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--algorithm", default="dqn")
    parser.add_argument("--timesteps", type=int, default=50_000)
    args = parser.parse_args()

    path = train_for_company(args.company_id, args.dataset_id, args.algorithm, args.timesteps)
    print(f"Modelo guardado en: {path}")
