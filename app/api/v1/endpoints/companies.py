"""Alta y gestión de empresas clientes (tenants) de la plataforma."""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.models.company import Company
from app.schemas.common import CompanyCreate, CompanyOut, CompanyWithApiKey
from config import get_settings
from database import get_db
from rate_limit import client_ip, limiter
from security import assign_new_api_key, get_current_company

router = APIRouter(prefix="/companies", tags=["companies"])
settings = get_settings()


def _with_api_key(company: Company, api_key: str) -> CompanyWithApiKey:
    return CompanyWithApiKey(**CompanyOut.model_validate(company).model_dump(), api_key=api_key)


@router.post("", response_model=CompanyWithApiKey, status_code=201)
@limiter.limit(lambda: settings.REGISTRATION_RATE_LIMIT, key_func=client_ip)
def register_company(request: Request, payload: CompanyCreate, db: Session = Depends(get_db)):
    """Registra una nueva empresa y le entrega su API Key, que se muestra solo esta vez.
    La empresa usará esa key en el header X-API-Key para todos los demás endpoints."""
    existing = db.query(Company).filter(Company.contact_email == payload.contact_email).first()
    if existing:
        raise HTTPException(status_code=409, detail="La empresa ya está registrada")

    company = Company(name=payload.name, contact_email=payload.contact_email)
    api_key = assign_new_api_key(company)
    db.add(company)
    db.commit()
    db.refresh(company)
    return _with_api_key(company, api_key)


@router.get("/me", response_model=CompanyOut)
def get_my_company(company: Company = Depends(get_current_company)):
    """Devuelve los datos de la empresa autenticada (según el header X-API-Key)."""
    return company


@router.post("/me/rotate-key", response_model=CompanyWithApiKey)
@limiter.limit("10/hour", key_func=client_ip)
def rotate_api_key(
    request: Request,
    company: Company = Depends(get_current_company),
    db: Session = Depends(get_db),
):
    """Genera una API key nueva y deja inválida la actual de inmediato. La nueva se muestra
    solo en esta respuesta."""
    api_key = assign_new_api_key(company)
    db.commit()
    db.refresh(company)
    return _with_api_key(company, api_key)
