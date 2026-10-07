"""
Carga de datasets por empresa. Cada empresa sube un CSV con sus datos históricos
(hoy, el único caso que entrena algo es lead_match: leads históricos con su
resultado de conversión -- ver app/models/dataset.py sobre por qué dynamic_pricing
y smart_route no usan este endpoint). El archivo se guarda en disco y se registra
en la DB; luego el endpoint /train del caso correspondiente lo toma como fuente
para (re)entrenar el modelo de ESA empresa.

Este router solo se ocupa de las partes HTTP (auth, 404s, códigos de estado). El
guardado en disco, el hash de contenido, el conteo de filas y la regla de "dataset
activo" viven en app/services/dataset_ingestion.py, compartido con
scripts/seed_dataset.py (carga directa sin pasar por HTTP, para desarrollo/tests).
"""

import os

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from app.models.company import Company
from app.models.dataset import CaseType, Dataset
from app.schemas.common import DatasetOut
from app.services.dataset_ingestion import (
    DatasetSchemaError,
    activate_dataset,
    ingest_dataset,
    read_dataset_csv,
    replace_dataset,
)
from database import get_db
from rate_limit import limiter
from security import get_current_company

router = APIRouter(prefix="/datasets", tags=["datasets"])

ALLOWED_EXTENSIONS = (".csv", ".tsv", ".txt")


def _check_extension(file: UploadFile) -> None:
    # Case-insensitive: Windows/Excel a veces guarda "DATOS.CSV" en mayúsculas.
    if not (file.filename or "").lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .csv, .tsv o .txt")


def _get_owned_dataset(db: Session, company: Company, case: CaseType, dataset_id: int) -> Dataset:
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.company_id == company.id, Dataset.case == case)
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")
    return dataset


@router.post("/{case}", response_model=DatasetOut, status_code=201)
@limiter.limit("10/minute")
def upload_dataset(
    case: CaseType,
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    company: Company = Depends(get_current_company),
):
    """Sube un dataset (CSV) para un caso específico, asociado a la empresa autenticada.

    Si el contenido es idéntico (mismo hash) a un dataset ya subido para esa empresa
    y caso, NO se duplica: se reactiva el existente y se devuelve ese mismo registro.
    En cualquier otro caso se crea un dataset nuevo y queda como el "activo" para
    (empresa, caso) -- ver el campo `is_active` en la respuesta, y el endpoint
    `/activate` para volver a marcar como vigente un dataset anterior."""
    _check_extension(file)

    try:
        dataset, _created = ingest_dataset(db, company.id, case, file.filename, file.file, activate=True)
    except DatasetSchemaError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el CSV: {exc}") from exc

    return dataset


@router.put("/{case}/{dataset_id}", response_model=DatasetOut)
@limiter.limit("10/minute")
def replace_dataset_endpoint(
    case: CaseType,
    dataset_id: int,
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    company: Company = Depends(get_current_company),
):
    """Reemplaza el contenido de un dataset ya subido, sobre el mismo dataset_id (a
    diferencia de POST, que siempre crea uno nuevo o reactiva uno con el mismo
    contenido). Pensado para corregir un upload con datos malos sin dejar
    "huérfano" el dataset_id si ya hay un modelo entrenado que lo referencia en su
    metadata (ver model_registry.save_version)."""
    _check_extension(file)

    dataset = _get_owned_dataset(db, company, case, dataset_id)
    try:
        dataset = replace_dataset(db, dataset, file.filename, file.file, activate=True)
    except DatasetSchemaError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el CSV: {exc}") from exc

    return dataset


@router.post("/{case}/{dataset_id}/activate", response_model=DatasetOut)
def activate_dataset_endpoint(
    case: CaseType,
    dataset_id: int,
    db: Session = Depends(get_db),
    company: Company = Depends(get_current_company),
):
    """Marca este dataset como el vigente para (empresa, caso) -- por ejemplo, para
    volver a una versión anterior sin re-subir el archivo. No cambia el
    comportamiento de /train (sigue pidiendo dataset_id explícito): esto es
    metadata para que la empresa sepa, mirando GET /datasets/{case}, cuál es "el"
    dataset vigente entre varios que tenga subidos."""
    dataset = _get_owned_dataset(db, company, case, dataset_id)
    return activate_dataset(db, dataset)


@router.delete("/{case}/{dataset_id}", status_code=204)
def delete_dataset(
    case: CaseType,
    dataset_id: int,
    db: Session = Depends(get_db),
    company: Company = Depends(get_current_company),
):
    """Borra un dataset (archivo en disco + registro en la DB). Si era el dataset
    activo, (empresa, caso) queda simplemente sin dataset activo -- no se reactiva
    automáticamente otro, para no promover en silencio un dataset que la empresa ya
    había dejado de usar a propósito."""
    dataset = _get_owned_dataset(db, company, case, dataset_id)
    if os.path.exists(dataset.file_path):
        os.remove(dataset.file_path)
    db.delete(dataset)
    db.commit()


@router.get("/{case}", response_model=list[DatasetOut])
def list_datasets(
    case: CaseType,
    db: Session = Depends(get_db),
    company: Company = Depends(get_current_company),
):
    return (
        db.query(Dataset)
        .filter(Dataset.company_id == company.id, Dataset.case == case)
        .order_by(Dataset.created_at.desc())
        .all()
    )


@router.get("/{case}/{dataset_id}/preview")
def preview_dataset(
    case: CaseType,
    dataset_id: int,
    db: Session = Depends(get_db),
    company: Company = Depends(get_current_company),
):
    """Devuelve las primeras filas del dataset, útil para validar el mapeo de columnas en el frontend."""
    dataset = _get_owned_dataset(db, company, case, dataset_id)

    df = read_dataset_csv(dataset.file_path, nrows=20)
    return {"columns": list(df.columns), "rows": df.fillna("").to_dict(orient="records")}
