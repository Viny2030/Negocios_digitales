"""Modelo de Dataset subido por una empresa para un caso (1, 2 o 3).
El archivo físico (csv/parquet) se guarda en DATASETS_DIR y se referencia por path."""
import datetime as dt
import enum

from sqlalchemy import Column, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.core.database import Base


class CaseType(str, enum.Enum):
    case1_messaging = "case1_messaging"
    case2_pricing = "case2_pricing"
    case3_anomaly = "case3_anomaly"


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
    status = Column(Enum(DatasetStatus), default=DatasetStatus.uploaded)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    company = relationship("Company", back_populates="datasets")
