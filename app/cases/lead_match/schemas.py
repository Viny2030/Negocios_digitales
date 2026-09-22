"""
LeadMatch Engine — Scoring y Enrutamiento Inteligente de Prospectos.

Dos componentes algorítmicos separados, cada uno con su propio módulo:
  - scoring.py: probabilidad de conversión de cada lead (regresión logística
    entrenada con leads históricos de la empresa -- ver /train -- con
    heurística de respaldo si todavía no hay modelo entrenado).
  - matching.py: asignación ponderada lead-vendedor (algoritmo húngaro) que
    combina esa probabilidad con la tasa de cierre histórica y la
    especialidad de cada vendedor, respetando su capacidad de leads del
    período.
"""

from pydantic import BaseModel, Field, model_validator


class LeadFeatures(BaseModel):
    lead_id: str
    origin: str = Field(..., description="Canal de origen del lead, ej. 'paid_ads', 'organic', 'referral'")
    time_on_site_minutes: float = Field(..., ge=0)
    company_size: int = Field(..., ge=0, description="Cantidad de empleados de la empresa del lead")


class SalesRep(BaseModel):
    rep_id: str
    capacity: int = Field(..., ge=0, description="Cuántos leads puede tomar este vendedor en esta ronda")
    historical_close_rate: float = Field(
        default=0.5, ge=0, le=1, description="Tasa de cierre histórica del vendedor, la maneja el CRM del cliente"
    )
    specialties: list[str] = Field(
        default_factory=list, description="Orígenes de lead en los que el vendedor históricamente convierte mejor"
    )


class MatchRequest(BaseModel):
    leads: list[LeadFeatures] = Field(..., min_length=1)
    reps: list[SalesRep] = Field(..., min_length=1)
    specialty_bonus: float = Field(
        default=1.2,
        ge=1.0,
        description="Multiplicador aplicado cuando el origen del lead está en las specialties del rep",
    )

    @model_validator(mode="after")
    def _ids_unicos(self) -> "MatchRequest":
        lead_ids = [ld.lead_id for ld in self.leads]
        if len(lead_ids) != len(set(lead_ids)):
            raise ValueError("lead_id duplicado en leads")
        rep_ids = [r.rep_id for r in self.reps]
        if len(rep_ids) != len(set(rep_ids)):
            raise ValueError("rep_id duplicado en reps")
        return self


class LeadAssignment(BaseModel):
    lead_id: str
    rep_id: str
    conversion_probability: float
    match_score: float = Field(
        description="conversion_probability * afinidad del rep, lo que maximiza el matching"
    )


class MatchResponse(BaseModel):
    assignments: list[LeadAssignment]
    unmatched_leads: list[str] = Field(
        default_factory=list, description="Leads que quedaron sin vendedor por falta de capacidad total"
    )
    used_default_scoring: bool = Field(
        description="True si no había modelo de conversión entrenado y se usó la heurística de respaldo"
    )


class TrainRequest(BaseModel):
    dataset_id: int


class TrainResponse(BaseModel):
    status: str
    version_id: str
    metrics: dict
