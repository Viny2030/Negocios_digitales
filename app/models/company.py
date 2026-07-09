"""Modelo de Empresa (tenant). Cada empresa que usa la plataforma tiene su propio API key,
sus propios datasets y sus propios modelos entrenados por caso."""
import datetime as dt

from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.orm import relationship

from app.core.database import Base


class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    contact_email = Column(String(255), nullable=False)
    api_key = Column(String(128), unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    datasets = relationship("Dataset", back_populates="company", cascade="all, delete-orphan")
