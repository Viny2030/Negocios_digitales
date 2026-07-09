"""
Autenticación por API Key para empresas clientes de la plataforma.
Cada empresa recibe una api_key al registrarse (ver endpoints/companies.py).
Todos los endpoints de negocio (/case1, /case2, /case3, /datasets) requieren
el header X-API-Key y resuelven la empresa (tenant) a partir de él.
"""
import secrets

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.models.company import Company

settings = get_settings()
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)


def generate_api_key() -> str:
    return f"pk_{secrets.token_urlsafe(32)}"


def get_current_company(
    api_key: str | None = Security(api_key_header),
    db: Session = Depends(get_db),
) -> Company:
    """Resuelve la empresa dueña del request a partir del API key. Falla con 401 si no es válida."""
    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Falta X-API-Key")

    company = db.query(Company).filter(Company.api_key == api_key, Company.is_active.is_(True)).first()
    if not company:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API Key inválida")
    return company
