"""
Entrenamiento offline del Caso 6.2 (asignación de recursos en aceleradora) para una
empresa específica.

Uso manual:
    python -m scripts.train_case4 --company-id 1 --dataset-id 9 --algorithm ppo --timesteps 50000

Nota: el espacio de acciones de AcceleratorEnv es continuo (Box), así que solo PPO
está soportado (DQN requiere acciones discretas) — a diferencia de train_case1/2/3,
acá no se ofrece "dqn" como alternativa válida.
"""
import argparse
import os

import pandas as pd
from sqlalchemy.orm import Session

from app.cases.case4_accelerator.env import AcceleratorEnv
from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.dataset import Dataset

settings = get_settings()


def train_for_company(company_id: int, dataset_id: int, algorithm: str = "ppo", timesteps: int = 50_000) -> str:
    from stable_baselines3 import PPO

    if algorithm.lower() != "ppo":
        raise ValueError(
            "Case 4 (accelerator) usa un espacio de acciones continuo (Box): solo PPO está "
            "soportado. DQN requiere un espacio de acciones discreto."
        )

    db: Session = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.company_id == company_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} no encontrado para company_id={company_id}")
        historical_df = pd.read_csv(dataset.file_path)
    finally:
        db.close()

    env = AcceleratorEnv(historical_df=historical_df)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)

    out_dir = os.path.join(settings.MODELS_DIR, str(company_id), "case4_accelerator")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_ppo.zip")
    model.save(out_path)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--company-id", type=int, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--algorithm", default="ppo")
    parser.add_argument("--timesteps", type=int, default=50_000)
    args = parser.parse_args()

    path = train_for_company(args.company_id, args.dataset_id, args.algorithm, args.timesteps)
    print(f"Modelo guardado en: {path}")