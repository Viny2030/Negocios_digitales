"""
Carga un CSV local directo a storage/datasets + la tabla `datasets`, SIN pasar por
el endpoint HTTP (POST /api/v1/datasets/{case}). Pensado para desarrollo y tests con
los datasets sintéticos de synthetic_data/: evita levantar el server y hacer un
curl -F por cada CSV solo para tener algo con qué entrenar/probar.

Reusa exactamente la misma lógica que el endpoint (ver
app/services/dataset_ingestion.py) -- dedupe por hash, conteo de filas, inferencia
de columnas y la regla de "dataset activo" son una sola implementación, no dos
copiadas entre el router y este script.

Uso manual:
    python -m scripts.seed_dataset --company-id 1 --case lead_match \
        --file synthetic_data/lead_match.csv

    # Todos los datasets sintéticos disponibles para la empresa 1 de una sola vez
    # (hoy, solo lead_match -- ver CaseType en app/models/dataset.py):
    python -m scripts.seed_dataset --company-id 1 --all-synthetic
"""

import argparse
import logging
import os

from sqlalchemy.orm import Session

from app.models.dataset import CaseType
from app.services.dataset_ingestion import ingest_dataset
from database import SessionLocal

logger = logging.getLogger(__name__)

SYNTHETIC_DIR = "synthetic_data"


def seed_dataset(company_id: int, case: str, file_path: str, activate: bool = True) -> dict:
    case_enum = CaseType(case)

    db: Session = SessionLocal()
    try:
        with open(file_path, "rb") as f:
            dataset, created = ingest_dataset(
                db,
                company_id=company_id,
                case=case_enum,
                filename=os.path.basename(file_path),
                file_obj=f,
                activate=activate,
            )
    finally:
        db.close()

    return {
        "dataset_id": dataset.id,
        "created": created,
        "row_count": dataset.row_count,
        "is_active": dataset.is_active,
        "content_hash": dataset.content_hash,
    }


def seed_all_synthetic(company_id: int, synthetic_dir: str = SYNTHETIC_DIR) -> list[dict]:
    """Carga los CSVs de synthetic_data/ (uno por caso con dataset -- ver CaseType)
    para `company_id`, para dejar la empresa lista para entrenar sin pasos manuales."""
    results = []
    for case in CaseType:
        file_path = os.path.join(synthetic_dir, f"{case.value}.csv")
        if not os.path.exists(file_path):
            logger.warning("No se encontró %s, se omite %s.", file_path, case.value)
            continue
        result = seed_dataset(company_id, case.value, file_path)
        result["case"] = case.value
        results.append(result)
    return results


if __name__ == "__main__":
    import logging_config  # noqa: F401  (configura el formato/nivel de logging al correr como script)

    parser = argparse.ArgumentParser()
    parser.add_argument("--company-id", type=int, required=True)
    parser.add_argument("--case", choices=[c.value for c in CaseType])
    parser.add_argument("--file")
    parser.add_argument("--all-synthetic", action="store_true", help=f"Carga los 4 CSVs de {SYNTHETIC_DIR}/")
    parser.add_argument("--no-activate", action="store_true")
    args = parser.parse_args()

    if args.all_synthetic:
        for result in seed_all_synthetic(args.company_id):
            logger.info(
                "%s: dataset_id=%s %s row_count=%s is_active=%s",
                result["case"],
                result["dataset_id"],
                "creado" if result["created"] else "ya existía (mismo contenido), reactivado",
                result["row_count"],
                result["is_active"],
            )
    else:
        if not args.case or not args.file:
            parser.error("Sin --all-synthetic hacen falta --case y --file.")
        result = seed_dataset(args.company_id, args.case, args.file, activate=not args.no_activate)
        logger.info(
            "dataset_id=%s %s row_count=%s is_active=%s",
            result["dataset_id"],
            "creado" if result["created"] else "ya existía (mismo contenido), reactivado",
            result["row_count"],
            result["is_active"],
        )
