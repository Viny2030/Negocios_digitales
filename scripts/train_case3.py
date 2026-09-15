"""
Entrenamiento offline del Caso 3.3 (triaje de anomalías) para una empresa específica:
  1) Entrena el clasificador auxiliar de falso positivo (RandomForest) sobre tickets históricos.
  2) Entrena la política RL (DQN) de triaje sobre el entorno de cola simulado.

Uso manual:
    python -m scripts.train_case3 --company-id 1 --dataset-id 7 --timesteps 50000
"""

import argparse
import logging
import os
import tempfile

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

import model_registry
from app.cases.case3_anomaly.env import AnomalyTriageEnv
from app.models.dataset import Dataset
from config import get_settings
from database import SessionLocal

logger = logging.getLogger(__name__)
settings = get_settings()
CASE = "case3_anomaly"


def _train_fp_classifier(historical_df: pd.DataFrame, tmp_dir: str) -> str | None:
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

    tmp_path = os.path.join(tmp_dir, "fp_classifier.joblib")
    joblib.dump(clf, tmp_path)
    return tmp_path


def train_for_company(company_id: int, dataset_id: int, timesteps: int = 50_000) -> dict:
    from stable_baselines3 import DQN

    db: Session = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.company_id == company_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} no encontrado para company_id={company_id}")
        historical_df = pd.read_csv(dataset.file_path)
        # Se capturan antes de cerrar la sesión: quedan en `metrics` para poder trazar con qué
        # dataset exacto (id + hash de contenido) se entrenó esta versión de modelo/clasificador.
        dataset_content_hash = dataset.content_hash
        dataset_row_count = dataset.row_count
    finally:
        db.close()

    version_id = model_registry.new_version_id("dqn")
    out_dir = os.path.join(settings.MODELS_DIR, str(company_id), CASE)
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        classifier_tmp_path = _train_fp_classifier(historical_df, tmp_dir)

        env = AnomalyTriageEnv(historical_df=historical_df)
        model = DQN("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=timesteps)
        rl_tmp_path = os.path.join(tmp_dir, "model.zip")
        model.save(rl_tmp_path)

        artifacts = {"model.zip": rl_tmp_path}
        if classifier_tmp_path:
            artifacts["fp_classifier.joblib"] = classifier_tmp_path

        metrics = {
            "algorithm": "dqn",
            "timesteps": timesteps,
            "approved": True,
            "classifier_trained": classifier_tmp_path is not None,
            "dataset_id": dataset_id,
            "dataset_content_hash": dataset_content_hash,
            "dataset_row_count": dataset_row_count,
        }
        version_dir = model_registry.save_version(
            settings.MODELS_DIR,
            company_id,
            CASE,
            version_id,
            artifacts=artifacts,
            metrics=metrics,
        )

    model_registry.promote(settings.MODELS_DIR, company_id, CASE, version_id)

    return {
        "version_id": version_id,
        "version_dir": version_dir,
        "approved": True,
        "rl_model_path": os.path.join(version_dir, "model.zip"),
        "classifier_path": os.path.join(version_dir, "fp_classifier.joblib") if classifier_tmp_path else None,
    }


if __name__ == "__main__":
    import logging_config  # noqa: F401  (configura el formato/nivel de logging al correr como script)

    parser = argparse.ArgumentParser()
    parser.add_argument("--company-id", type=int, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=50_000)
    args = parser.parse_args()

    paths = train_for_company(args.company_id, args.dataset_id, args.timesteps)
    logger.info("Modelos guardados: %s", paths)
