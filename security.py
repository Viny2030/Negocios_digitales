"""
Autenticación por API Key para empresas clientes de la plataforma.
Cada empresa recibe una api_key al registrarse (ver endpoints/companies.py).
Todos los endpoints de negocio (/dynamic-pricing, /smart-route, /lead-match, /datasets) requieren
el header X-API-Key y resuelven la empresa (tenant) a partir de él.

En la base se guarda solo el hash de la key, nunca la key: una copia filtrada de la base
no sirve para autenticarse. Por eso la key se muestra una sola vez, al crearla o rotarla.
Alcanza con SHA-256 sin sal porque la key es un token aleatorio de 256 bits, no una
contraseña elegida por una persona.
"""

import hashlib
import secrets

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from app.models.company import Company
from config import get_settings
from database import get_db

settings = get_settings()
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)

API_KEY_PREFIX_LENGTH = 11  # "pk_" + 8 caracteres: identifica la key sin permitir usarla


def generate_api_key() -> str:
    return f"pk_{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def assign_new_api_key(company: Company) -> str:
    """Le asigna a `company` una key nueva, invalidando la anterior si la había, y la
    devuelve en claro. Es la única vez que queda disponible."""
    api_key = generate_api_key()
    company.api_key_hash = hash_api_key(api_key)
    company.api_key_prefix = api_key[:API_KEY_PREFIX_LENGTH]
    return api_key


def get_current_company(
    api_key: str | None = Security(api_key_header),
    db: Session = Depends(get_db),
) -> Company:
    """Resuelve la empresa dueña del request a partir del API key. Falla con 401 si no es válida."""
    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Falta X-API-Key")

    company = (
        db.query(Company)
        .filter(Company.api_key_hash == hash_api_key(api_key), Company.is_active.is_(True))
        .first()
    )
    if not company:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API Key inválida")
    return company
