"""Alta y gestión de empresas clientes (tenants) de la plataforma."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import generate_api_key, get_current_company
from app.models.company import Company
from app.schemas.common import CompanyCreate, CompanyOut

router = APIRouter(prefix="/companies", tags=["companies"])


@router.post("", response_model=CompanyOut, status_code=201)
def register_company(payload: CompanyCreate, db: Session = Depends(get_db)):
    """Registra una nueva empresa y le entrega su API Key.
    La empresa usará esa key en el header X-API-Key para todos los demás endpoints."""
    existing = db.query(Company).filter(Company.contact_email == payload.contact_email).first()
    if existing:
        raise HTTPException(status_code=409, detail="La empresa ya está registrada")

    company = Company(
        name=payload.name,
        contact_email=payload.contact_email,
        api_key=generate_api_key(),
    )
    db.add(company)
    db.commit()
    db.refresh(company)
    return company


@router.get("/me", response_model=CompanyOut)
def get_my_company(company: Company = Depends(get_current_company)):
    """Devuelve los datos de la empresa autenticada (según el header X-API-Key)."""
    return company
