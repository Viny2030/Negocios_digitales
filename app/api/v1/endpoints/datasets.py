"""
Carga de datasets por empresa. Cada empresa sube un CSV con SUS datos históricos
(por ejemplo: campañas de marketing para case1, transacciones/precios para case2,
alertas resueltas para case3). El archivo se guarda en disco y se registra en la DB;
luego los scripts de entrenamiento (scripts/train_case*.py) o el endpoint /train
de cada caso lo toman como fuente para (re)entrenar el modelo de ESA empresa.
"""
import json
import os
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.core.security import get_current_company
from app.models.company import Company
from app.models.dataset import CaseType, Dataset, DatasetStatus
from app.schemas.common import DatasetOut

router = APIRouter(prefix="/datasets", tags=["datasets"])
settings = get_settings()


@router.post("/{case}", response_model=DatasetOut, status_code=201)
def upload_dataset(
    case: CaseType,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    company: Company = Depends(get_current_company),
):
    """Sube un dataset (CSV) para un caso específico, asociado a la empresa autenticada."""
    if not file.filename.endswith((".csv", ".tsv")):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .csv o .tsv")

    company_dir = os.path.join(settings.DATASETS_DIR, str(company.id), case.value)
    os.makedirs(company_dir, exist_ok=True)

    stored_name = f"{uuid.uuid4().hex}_{file.filename}"
    dest_path = os.path.join(company_dir, stored_name)
    with open(dest_path, "wb") as f:
        f.write(file.file.read())

    try:
        df = pd.read_csv(dest_path, nrows=5000)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el CSV: {exc}") from exc

    dataset = Dataset(
        company_id=company.id,
        case=case,
        name=file.filename,
        file_path=dest_path,
        row_count=len(df),
        columns_schema=json.dumps({c: str(t) for c, t in df.dtypes.items()}),
        status=DatasetStatus.validated,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


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
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.company_id == company.id, Dataset.case == case)
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    df = pd.read_csv(dataset.file_path, nrows=20)
    return {"columns": list(df.columns), "rows": df.fillna("").to_dict(orient="records")}
