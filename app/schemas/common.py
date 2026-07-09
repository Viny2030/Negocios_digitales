"""Esquemas Pydantic compartidos: empresas y datasets."""
import datetime as dt

from pydantic import BaseModel, EmailStr

from app.models.dataset import CaseType, DatasetStatus


class CompanyCreate(BaseModel):
    name: str
    contact_email: EmailStr


class CompanyOut(BaseModel):
    id: int
    name: str
    contact_email: EmailStr
    api_key: str
    is_active: bool
    created_at: dt.datetime

    class Config:
        from_attributes = True


class DatasetOut(BaseModel):
    id: int
    company_id: int
    case: CaseType
    name: str
    row_count: int
    status: DatasetStatus
    created_at: dt.datetime

    class Config:
        from_attributes = True
