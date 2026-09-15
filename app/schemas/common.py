"""Esquemas Pydantic compartidos: empresas y datasets."""

import datetime as dt

from pydantic import BaseModel, ConfigDict, EmailStr

from app.models.dataset import CaseType, DatasetStatus


class CompanyCreate(BaseModel):
    name: str
    contact_email: EmailStr


class CompanyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    contact_email: EmailStr
    api_key: str
    is_active: bool
    created_at: dt.datetime


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    company_id: int
    case: CaseType
    name: str
    row_count: int
    status: DatasetStatus
    # Dataset vigente para (company_id, case) -- ver app/services/dataset_ingestion.py.
    is_active: bool
    created_at: dt.datetime
    # Solo se completa si el dataset fue reemplazado in-place (PUT /datasets/{case}/{id}).
    updated_at: dt.datetime | None = None
