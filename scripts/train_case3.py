"""
Entrenamiento offline del Caso 3.3 (triaje de anomalías) para una empresa específica:
  1) Entrena el clasificador auxiliar de falso positivo (RandomForest) sobre tickets históricos.
  2) Entrena la política RL (DQN) de triaje sobre el entorno de cola simulado.

Uso manual:
    python -m scripts.train_case3 --company-id 1 --dataset-id 7 --timesteps 50000
"""
import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from app.cases.case3_anomaly.env import AnomalyTriageEnv
from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.dataset import Dataset

settings = get_settings()


def _train_fp_classifier(historical_df: pd.DataFrame, out_dir: str) -> str | None:
    """Espera columnas: z_score, pct_change, historical_similar_count, estimated_financial_impact,
    label (1 = false_positive, 0 = real_incident). Si el dataset no trae `label`, se omite este paso."""
    if "label" not in historical_df.columns:
        return None

    feature_cols = ["z_score", "pct_change", "historical_similar_count", "estimated_financial_impact"]
    available_cols = [c for c in feature_cols if c in historical_df.columns]
    X = historical_df[available_cols].fillna(0)
    y = historical_df["label"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    out_path = os.path.join(out_dir, "fp_classifier.joblib")
    joblib.dump(clf, out_path)
    return out_path


def train_for_company(company_id: int, dataset_id: int, timesteps: int = 50_000) -> dict:
    from stable_baselines3 import DQN

    db: Session = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.company_id == company_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} no encontrado para company_id={company_id}")
        historical_df = pd.read_csv(dataset.file_path)
    finally:
        db.close()

    out_dir = os.path.join(settings.MODELS_DIR, str(company_id), "case3_anomaly")
    os.makedirs(out_dir, exist_ok=True)

    classifier_path = _train_fp_classifier(historical_df, out_dir)

    env = AnomalyTriageEnv(historical_df=historical_df)
    model = DQN("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    rl_path = os.path.join(out_dir, "model.zip")
    model.save(rl_path)

    return {"rl_model_path": rl_path, "classifier_path": classifier_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--company-id", type=int, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=50_000)
    args = parser.parse_args()

    paths = train_for_company(args.company_id, args.dataset_id, args.timesteps)
    print(f"Modelos guardados: {paths}")
