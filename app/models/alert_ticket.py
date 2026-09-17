"""Registro de tickets creados por el Caso 3.3 (triaje de anomalías), usado para idempotencia:
un webhook de alerta reintentado (mismo `alert_id`) no debe generar un segundo ticket en la
herramienta de gestión de la empresa (Jira/Opsgenie/Zendesk). Ver
app/cases/case3_anomaly/service.py::_create_ticket_if_needed."""

import datetime as dt

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship

from database import Base


class AlertTicket(Base):
    __tablename__ = "alert_tickets"
    __table_args__ = (UniqueConstraint("company_id", "alert_id", name="uq_alert_tickets_company_alert"),)

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    # id de la alerta tal como llega en el payload del webhook (Datadog/Looker/BI), único por
    # empresa junto con company_id (ver UniqueConstraint arriba).
    alert_id = Column(String(255), nullable=False)
    action = Column(String(50), nullable=False)
    ticket_created = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=lambda: dt.datetime.now(dt.UTC))

    company = relationship("Company")
