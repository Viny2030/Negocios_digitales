"""
Lógica de ingesta de datasets: guardado en disco (en streaming, sin cargar el
archivo entero en RAM), hash de contenido, conteo de filas, validación de que el
CSV trae columnas del caso que dice ser, inferencia de columnas y la regla de
"dataset activo" por (empresa, caso).

Antes esta lógica vivía entera dentro de app/api/v1/endpoints/datasets.py. Se separó
a este módulo por dos motivos:

1. Permite un segundo camino de carga sin pasar por HTTP (ver scripts/seed_dataset.py),
   pensado para desarrollo/tests con los CSVs de synthetic_data/, sin tener que levantar
   el server y hacer un curl -F por cada uno -- ambos caminos llaman a las mismas
   funciones de acá, así que no hay dos implementaciones del mismo negocio para
   mantener sincronizadas.
2. El endpoint original no tenía forma de "reemplazar" un dataset ya subido (cada
   POST creaba una fila y un archivo nuevos, sin tocar los anteriores) ni de saber
   cuál era el vigente entre varios uploads de una misma empresa/caso. `ingest_dataset`,
   `replace_dataset` y `activate_dataset` resuelven eso con una sola fuente de verdad.
"""

import csv
import datetime as dt
import hashlib
import json
import os
import tempfile
import uuid

import pandas as pd
from sqlalchemy.orm import Session

from app.models.dataset import CaseType, Dataset, DatasetStatus
from config import get_settings

settings = get_settings()

# Columnas "identificatorias" de cada caso: no hace falta que estén TODAS (el resto del
# pipeline ya es tolerante -- ver record_to_* en cada encoding.py, que usan defaults para
# lo que falte), pero si el CSV subido no trae NINGUNA de las de su propio caso, es casi
# seguro que se subió el dataset equivocado (p.ej. el de case1 al endpoint de case2) y
# antes quedaba igual aceptado como "validated". Ver MIN_MATCHING_COLUMNS.
EXPECTED_COLUMNS: dict[CaseType, set[str]] = {
    CaseType.lead_match: {
        "lead_id",
        "origin",
        "time_on_site_minutes",
        "company_size",
        "rep_id",
        "converted",
    },
}
MIN_MATCHING_COLUMNS = 2


class DatasetSchemaError(ValueError):
    """El CSV se pudo leer, pero sus columnas no coinciden con las esperadas para el
    caso indicado (probablemente se subió el dataset de otro caso)."""


def _peek_header_columns(file_obj) -> list[str]:
    """Lee solo la primera línea de `file_obj` (sin consumirlo: deja el cursor donde
    estaba) para validar columnas antes de escribir nada a disco."""
    pos = file_obj.tell()
    line = file_obj.readline()
    file_obj.seek(pos)
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    return next(csv.reader([line]), [])


def _validate_expected_columns(case: CaseType, columns: list[str]) -> None:
    expected = EXPECTED_COLUMNS.get(case)
    if not expected:
        return
    overlap = expected & set(columns)
    if len(overlap) < MIN_MATCHING_COLUMNS:
        raise DatasetSchemaError(
            f"El CSV no parece un dataset de {case.value}: no trae ninguna de las columnas "
            f"esperadas para ese caso ({', '.join(sorted(expected))}). ¿Se subió el archivo "
            "de otro caso por error?"
        )


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stream_to_path(file_obj, dest_path: str, chunk_size: int = 1024 * 1024) -> tuple[str, int]:
    """Escribe `file_obj` en `dest_path` calculando el hash sha256 y contando filas de
    datos (líneas menos 1 por el header) en la misma pasada, sin cargar el archivo
    entero en memoria. Antes `upload_dataset` hacía `file.file.read()` completo a RAM
    antes de guardarlo -- con un dataset grande de una empresa real eso puede tumbar
    el proceso. Devuelve (content_hash, row_count)."""
    hasher = hashlib.sha256()
    newline_count = 0
    last_byte = b""
    with open(dest_path, "wb") as out:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            hasher.update(chunk)
            newline_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
            out.write(chunk)

    # Si el archivo no termina en salto de línea, esa última línea igual cuenta (mismo
    # criterio que la iteración línea a línea de Python sobre un archivo abierto).
    total_lines = newline_count + (1 if last_byte and last_byte != b"\n" else 0)
    row_count = max(total_lines - 1, 0)
    return hasher.hexdigest(), row_count


def infer_columns_schema(path: str, nrows: int = 5000) -> dict[str, str]:
    """Lee solo una muestra para inferir tipos de columna -- alcanza para
    `columns_schema` y es lo único para lo que hace falta pandas acá; el conteo de
    filas real sale de `_stream_to_path`, no de esto."""
    df = pd.read_csv(path, nrows=nrows)
    return {c: str(t) for c, t in df.dtypes.items()}


def _company_case_dir(company_id: int, case: CaseType) -> str:
    case_value = case.value if isinstance(case, CaseType) else case
    company_dir = os.path.join(settings.DATASETS_DIR, str(company_id), case_value)
    os.makedirs(company_dir, exist_ok=True)
    return company_dir


def _deactivate_siblings(db: Session, company_id: int, case: CaseType, except_id: int | None = None) -> None:
    query = db.query(Dataset).filter(
        Dataset.company_id == company_id,
        Dataset.case == case,
        Dataset.is_active.is_(True),
    )
    if except_id is not None:
        query = query.filter(Dataset.id != except_id)
    for sibling in query.all():
        sibling.is_active = False


def activate_dataset(db: Session, dataset: Dataset) -> Dataset:
    """Marca `dataset` como el vigente para su (company_id, case), desactivando
    cualquier otro dataset que estuviera activo para esa misma combinación. Es
    intencional que esto NO cambie qué usa `/train`: ese endpoint sigue pidiendo
    `dataset_id` explícito (ver TrainRequest de cada caso); esto es para que la
    empresa pueda ver/fijar cuál es "el" dataset vigente sin tener que recordar IDs."""
    _deactivate_siblings(db, dataset.company_id, dataset.case, except_id=dataset.id)
    dataset.is_active = True
    db.commit()
    db.refresh(dataset)
    return dataset


def ingest_dataset(
    db: Session,
    company_id: int,
    case: CaseType,
    filename: str,
    file_obj,
    activate: bool = True,
) -> tuple[Dataset, bool]:
    """Guarda el contenido de `file_obj` (cualquier objeto con `.read(n)` sync y
    seekable -- `UploadFile.file` del endpoint, o un archivo local abierto en
    `scripts/seed_dataset.py`) como dataset de `case` para `company_id`, en
    streaming. Devuelve `(dataset, created)`.

    Valida primero que el CSV traiga columnas de ESE caso (ver EXPECTED_COLUMNS):
    si no, corta antes de escribir nada a disco. Si el contenido es idéntico (mismo
    sha256) a un dataset ya subido para esa empresa y caso, tampoco se duplica: se
    descarta el archivo temporal, se reactiva el existente (si `activate=True`) y
    se devuelve con `created=False`."""
    _validate_expected_columns(case, _peek_header_columns(file_obj))

    company_dir = _company_case_dir(company_id, case)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=company_dir, prefix=".upload_")
    os.close(tmp_fd)
    try:
        content_hash, row_count = _stream_to_path(file_obj, tmp_path)

        existing = (
            db.query(Dataset)
            .filter(
                Dataset.company_id == company_id,
                Dataset.case == case,
                Dataset.content_hash == content_hash,
            )
            .order_by(Dataset.created_at.desc())
            .first()
        )
        if existing:
            os.remove(tmp_path)
            if activate and not existing.is_active:
                existing = activate_dataset(db, existing)
            return existing, False

        columns_schema = infer_columns_schema(tmp_path)
        stored_name = f"{uuid.uuid4().hex}_{filename}"
        dest_path = os.path.join(company_dir, stored_name)
        os.replace(tmp_path, dest_path)  # rename: mismo filesystem, tmp_path ya vive en company_dir
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    dataset = Dataset(
        company_id=company_id,
        case=case,
        name=filename,
        file_path=dest_path,
        row_count=row_count,
        columns_schema=json.dumps(columns_schema),
        content_hash=content_hash,
        status=DatasetStatus.validated,
        is_active=False,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    if activate:
        dataset = activate_dataset(db, dataset)

    return dataset, True


def replace_dataset(db: Session, dataset: Dataset, filename: str, file_obj, activate: bool = True) -> Dataset:
    """Reemplaza el contenido de un dataset YA EXISTENTE, sobre el mismo `dataset.id`
    y el mismo `file_path`, también en streaming. A diferencia de `ingest_dataset`
    (que siempre agrega una fila nueva salvo que el contenido sea idéntico a uno
    previo), esto es un 'sobreescribir' real: cualquier referencia externa al
    dataset_id -- por ejemplo, la metadata de un modelo ya entrenado con él, ver
    scripts/train_case*.py -- sigue apuntando al mismo registro, ahora con el
    contenido nuevo. Valida columnas contra el `case` ya guardado en el dataset."""
    _validate_expected_columns(dataset.case, _peek_header_columns(file_obj))

    company_dir = os.path.dirname(dataset.file_path)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=company_dir, prefix=".upload_")
    os.close(tmp_fd)
    try:
        content_hash, row_count = _stream_to_path(file_obj, tmp_path)
        columns_schema = infer_columns_schema(tmp_path)
        os.replace(tmp_path, dataset.file_path)  # sobreescribe atómicamente el archivo existente
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    dataset.content_hash = content_hash
    dataset.row_count = row_count
    dataset.columns_schema = json.dumps(columns_schema)
    dataset.name = filename
    dataset.status = DatasetStatus.validated
    dataset.updated_at = dt.datetime.now(dt.UTC)
    db.commit()
    db.refresh(dataset)

    if activate and not dataset.is_active:
        dataset = activate_dataset(db, dataset)

    return dataset
