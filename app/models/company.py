"""Modelo de Empresa (tenant). Cada empresa que usa la plataforma tiene su propio API key,
sus propios datasets y sus propios modelos entrenados por caso."""

import datetime as dt

from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.orm import relationship

from database import Base


class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    contact_email = Column(String(255), nullable=False)
    # Solo el hash de la API key, nunca la key en claro (ver security.py).
    api_key_hash = Column(String(64), unique=True, index=True, nullable=False)
    api_key_prefix = Column(String(16), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: dt.datetime.now(dt.UTC))

    datasets = relationship("Dataset", back_populates="company", cascade="all, delete-orphan")
