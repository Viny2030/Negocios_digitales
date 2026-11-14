"""
DynamicPricing API — Motor de Fijación Dinámica de Precios.

Contrato de negocio: el cliente (tienda de e-commerce) envía, por SKU, su costo
base, stock actual y los eventos de compra recientes (precio al que se vendió,
cantidad vendida) — la plataforma devuelve el precio recomendado en el momento,
sin necesidad de un paso de entrenamiento previo por separado (ver
elasticity.py: la elasticidad precio-demanda se estima on-the-fly a partir de
`purchase_events` en cada request). Esto es intencional y refleja el I/O
declarado en la propuesta: "el cliente envía vía API su stock, costo base y
eventos de compra; la plataforma devuelve el precio recomendado por SKU en
tiempo real" — no un ciclo async de upload-dataset → /train → esperar.
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class PricingObjective(str, Enum):
    margin = "margin"  # maximizar beneficio esperado (precio - costo) * demanda
    volume = "volume"  # maximizar volumen esperado dentro de las restricciones


class PurchaseEvent(BaseModel):
    """Una observación histórica de venta a un precio dado. Sirve de insumo para
    estimar la elasticidad precio-demanda del SKU (ver elasticity.py)."""

    price: float = Field(..., gt=0)
    quantity_sold: int = Field(..., ge=0)


class SkuPricingRequest(BaseModel):
    sku_id: str
    base_cost: float = Field(..., gt=0, description="Costo unitario del SKU")
    current_price: float = Field(..., gt=0)
    stock: int = Field(
        ...,
        ge=0,
        description="Unidades disponibles. Limita la cantidad esperada: no se puede vender más que esto. "
        "Se compara con la demanda estimada, así que debe referirse al mismo período que purchase_events.",
    )
    purchase_events: list[PurchaseEvent] = Field(
        default_factory=list,
        max_length=5000,
        description="Historial reciente de (precio, cantidad vendida). Con pocas "
        "observaciones o precios sin variación no se puede estimar la demanda: el motor no "
        "recomienda un cambio de precio (ver used_default_elasticity).",
    )
    competitor_price: float | None = Field(
        default=None,
        gt=0,
        description="Precio del competidor. Pone un techo según max_competitor_premium_pct.",
    )
    max_competitor_premium_pct: float = Field(
        default=0.10,
        ge=0,
        description="Cuánto por encima de competitor_price se puede vender. El margen mínimo y la "
        "variación diaria tienen prioridad sobre este techo.",
    )
    min_margin_pct: float = Field(
        default=0.15, ge=0, lt=1, description="Margen mínimo exigido sobre el costo base"
    )
    max_daily_variation_pct: float = Field(
        default=0.05, gt=0, le=1, description="Variación máxima permitida respecto al precio actual"
    )

    @field_validator("sku_id")
    @classmethod
    def _sku_id_no_vacio(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("sku_id no puede estar vacío")
        return v


class PricingRequest(BaseModel):
    skus: list[SkuPricingRequest] = Field(..., min_length=1, max_length=500)
    objective: PricingObjective = PricingObjective.margin

    @model_validator(mode="after")
    def _check_unique_sku_ids(self) -> "PricingRequest":
        seen: set[str] = set()
        duplicates: set[str] = set()
        for s in self.skus:
            if s.sku_id in seen:
                duplicates.add(s.sku_id)
            seen.add(s.sku_id)
        if duplicates:
            raise ValueError(f"sku_id duplicado(s) en skus: {sorted(duplicates)}")
        return self


class SkuPricingResult(BaseModel):
    sku_id: str
    recommended_price: float
    expected_quantity: float | None = Field(
        description="Unidades esperadas, limitadas por el stock, en el período de purchase_events. "
        "None si no hubo datos para estimar la demanda."
    )
    expected_margin_pct: float = Field(description="Ganancia como proporción del precio")
    expected_margin_on_cost_pct: float = Field(
        description="Ganancia como proporción del costo: es la que se compara con min_margin_pct"
    )
    expected_profit: float | None = Field(
        description="Ganancia esperada. None si no hubo datos para estimar la demanda."
    )
    elasticity: float | None = Field(
        description="Coeficiente b de la regresión log-log estimada (quantity ~ price^b). "
        "None si no hubo datos propios suficientes."
    )
    used_default_elasticity: bool = Field(
        description="True si no se pudo estimar la demanda. El precio no es una recomendación "
        "fundada: se mantiene el actual, ajustado solo si rompe una regla."
    )
    constraints_applied: list[str] = Field(default_factory=list)


class PricingResponse(BaseModel):
    results: list[SkuPricingResult]
