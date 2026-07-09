"""
Caso 2.1 — Política de pricing dinámico con restricciones de equidad.

Estado (S): producto, segmento, contexto, demanda, inventario, precios competencia.
Acciones (A): precio continuo dentro de banda permitida (multiplicador 1.0x-2.5x sobre costo).
Recompensa (R): revenue neto - penalización de margen - penalización de equidad (Demographic Parity).
Caso canónico del Anexo I (H1, OP1, OP2).
"""
from pydantic import BaseModel, Field


class PriceRequest(BaseModel):
    product_id: str
    base_cost: float = Field(..., gt=0, description="Costo marginal / piso de precio")
    user_segment: str
    inventory_level_pct: float = Field(..., ge=0, le=100)
    competitor_avg_price: float | None = None
    demand_index: float = Field(default=0.5, ge=0, le=1)
    min_margin_pct: float = Field(default=0.10, description="Margen mínimo exigido por finanzas")
    max_multiplier: float = Field(default=2.5)
    min_multiplier: float = Field(default=1.0)


class PriceResponse(BaseModel):
    product_id: str
    final_price: float
    raw_rl_price: float
    constraints_applied: list[str]
    latency_ms: float


class TrainRequest(BaseModel):
    dataset_id: int
    algorithm: str = Field(default="ppo", description="ppo | dqn")
    timesteps: int = 50_000
    fairness_penalty_weight: float = 1.0
