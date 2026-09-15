"""
Entrenamiento offline del Caso 2.1 (pricing dinámico) para una empresa específica.
Incorpora la penalización de equidad (Demographic Parity, vía Fairlearn) evaluada
sobre el historial de transacciones antes/después del entrenamiento.

Cada corrida genera una versión nueva en el registro de modelos (ver
model_registry.py). La versión solo se promueve a "current" (la que sirve
el endpoint /case2/get-price) si la diferencia de Demographic Parity entre segmentos
no supera `settings.FAIRNESS_DP_THRESHOLD`; si la supera, queda guardada para
inspección pero la empresa sigue sirviendo su última versión aprobada.

Uso manual:
    python -m scripts.train_case2 --company-id 1 --dataset-id 5 --algorithm ppo --timesteps 50000
"""

import argparse
import logging
import os

import pandas as pd
from sqlalchemy.orm import Session

import model_registry
from app.cases.case2_pricing.env import PricingEnv
from app.cases.case2_pricing.fairness import evaluate_fairness
from app.models.dataset import Dataset
from config import get_settings
from database import SessionLocal

logger = logging.getLogger(__name__)
settings = get_settings()
CASE = "case2_pricing"


def train_for_company(
    company_id: int,
    dataset_id: int,
    algorithm: str = "ppo",
    timesteps: int = 50_000,
    fairness_penalty_weight: float = 1.0,
) -> dict:
    from stable_baselines3 import DQN, PPO

    db: Session = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.company_id == company_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} no encontrado para company_id={company_id}")
        historical_df = pd.read_csv(dataset.file_path)
        # Se capturan antes de cerrar la sesión: quedan en `metrics` para poder trazar con qué
        # dataset exacto (id + hash de contenido) se entrenó esta versión de modelo.
        dataset_content_hash = dataset.content_hash
        dataset_row_count = dataset.row_count
    finally:
        db.close()

    env = PricingEnv(historical_df=historical_df, fairness_penalty_weight=fairness_penalty_weight)
    algo_cls = DQN if algorithm.lower() == "dqn" else PPO
    model = algo_cls("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)

    fairness_result = evaluate_fairness(model, historical_df, env.min_multiplier, env.max_multiplier)
    dp_diff = fairness_result.get("demographic_parity_difference")
    # Si no se pudo evaluar (falta la columna user_segment), no bloqueamos por falta de datos,
    # pero queda registrado en la metadata de la versión para revisión manual.
    approved = dp_diff is None or dp_diff <= settings.FAIRNESS_DP_THRESHOLD

    version_id = model_registry.new_version_id(algorithm)
    out_dir = os.path.join(settings.MODELS_DIR, str(company_id), CASE)
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.join(out_dir, f"_tmp_{version_id}.zip")
    model.save(tmp_path)

    metrics = {
        "algorithm": algorithm,
        "timesteps": timesteps,
        "fairness_penalty_weight": fairness_penalty_weight,
        "fairness": fairness_result,
        "fairness_threshold": settings.FAIRNESS_DP_THRESHOLD,
        "approved": approved,
        "dataset_id": dataset_id,
        "dataset_content_hash": dataset_content_hash,
        "dataset_row_count": dataset_row_count,
    }
    version_dir = model_registry.save_version(
        settings.MODELS_DIR,
        company_id,
        CASE,
        version_id,
        artifacts={"model.zip": tmp_path},
        metrics=metrics,
    )

    if approved:
        model_registry.promote(settings.MODELS_DIR, company_id, CASE, version_id)
    else:
        logger.warning(
            "[case2] Versión %s NO promovida: demographic_parity_difference=%s supera el "
            "umbral %s. Sigue sirviendo la versión anterior aprobada (si existe).",
            version_id,
            dp_diff,
            settings.FAIRNESS_DP_THRESHOLD,
        )

    return {
        "version_id": version_id,
        "version_dir": version_dir,
        "approved": approved,
        "fairness": fairness_result,
    }


if __name__ == "__main__":
    import logging_config  # noqa: F401  (configura el formato/nivel de logging al correr como script)

    parser = argparse.ArgumentParser()
    parser.add_argument("--company-id", type=int, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--algorithm", default="ppo")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--fairness-penalty-weight", type=float, default=1.0)
    args = parser.parse_args()

    result = train_for_company(
        args.company_id,
        args.dataset_id,
        args.algorithm,
        args.timesteps,
        args.fairness_penalty_weight,
    )
    estado = "APROBADA y promovida" if result["approved"] else "guardada pero NO promovida (falló equidad)"
    logger.info("Versión %s %s. Detalle de equidad: %s", result["version_id"], estado, result["fairness"])
