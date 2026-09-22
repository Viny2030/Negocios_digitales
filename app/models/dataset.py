"""Modelo de Dataset subido por una empresa para un caso.
El archivo físico (csv/parquet) se guarda en DATASETS_DIR y se referencia por path.

AGREGADO 2026-09-22: se reemplazaron los 4 casos originales (mensajería, pricing RL,
anomalías, acelerador) por los 3 productos de la plataforma actual. Solo `lead_match`
(LeadMatch Engine) tiene entrada acá: es el único de los 3 que entrena algo (el
scoring de conversión, ver app/cases/lead_match/). `dynamic_pricing` no necesita
dataset: los eventos de compra se mandan inline en cada request (ver el docstring de
app/cases/dynamic_pricing/schemas.py). `smart_route` tampoco: es una optimización
combinatoria stateless resuelta por request (ver app/cases/smart_route/), no entrena
ni versiona ningún modelo."""

import datetime as dt
import enum

from sqlalchemy import Boolean, Column, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from database import Base


class CaseType(str, enum.Enum):
    lead_match = "lead_match"


class DatasetStatus(str, enum.Enum):
    uploaded = "uploaded"
    validated = "validated"
    training = "training"
    ready = "ready"
    error = "error"


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    case = Column(Enum(CaseType), nullable=False)
    name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    row_count = Column(Integer, default=0)
    columns_schema = Column(Text, nullable=True)  # JSON con nombres/tipos de columnas detectadas
    # sha256 del contenido subido. Permite detectar re-uploads idénticos (mismo archivo subido
    # dos veces) sin comparar los CSVs byte a byte, y sin acumular copias/duplicados en la DB
    # ni en disco cada vez que una empresa vuelve a subir el mismo dataset. Ver
    # app/services/dataset_ingestion.py.
    content_hash = Column(String(64), nullable=True, index=True)
    # Dataset "vigente" para (company_id, case). A diferencia del modelo (ver model_registry.py,
    # que versiona con un puntero current.json en disco), acá el puntero de "activo" vive como
    # columna: solo puede haber un dataset activo por (company_id, case) a la vez, mantenido por
    # app/services/dataset_ingestion.py (activate_dataset / _deactivate_siblings). No implica que
    # /train lo use automáticamente -- ese endpoint sigue pidiendo dataset_id explícito -- es
    # sobre todo para que la empresa vea, en GET /datasets/{case}, cuál es "el" dataset vigente
    # entre varios subidos.
    is_active = Column(Boolean, default=False, nullable=False)
    status = Column(Enum(DatasetStatus), default=DatasetStatus.uploaded)
    created_at = Column(DateTime, default=lambda: dt.datetime.now(dt.UTC))
    # Se completa solo cuando el dataset se reemplaza in-place (PUT /datasets/{case}/{id}, ver
    # replace_dataset en app/services/dataset_ingestion.py). Un dataset nunca tocado por un
    # reemplazo queda con updated_at=None.
    updated_at = Column(DateTime, nullable=True)

    company = relationship("Company", back_populates="datasets")
