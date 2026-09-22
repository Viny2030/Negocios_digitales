"""Orquesta, por SKU: ajustar la elasticidad a partir de los eventos de compra
enviados y resolver el precio óptimo dentro de las restricciones de negocio."""

from app.cases.dynamic_pricing.elasticity import fit_elasticity
from app.cases.dynamic_pricing.optimizer import optimize_price
from app.cases.dynamic_pricing.schemas import (
    PricingRequest,
    PricingResponse,
    SkuPricingRequest,
    SkuPricingResult,
)


def _price_sku(sku: SkuPricingRequest, objective) -> SkuPricingResult:
    events = [(e.price, e.quantity_sold) for e in sku.purchase_events]
    elasticity = fit_elasticity(events)

    optimization = optimize_price(
        elasticity=elasticity,
        base_cost=sku.base_cost,
        current_price=sku.current_price,
        min_margin_pct=sku.min_margin_pct,
        max_daily_variation_pct=sku.max_daily_variation_pct,
        objective=objective,
    )

    price = optimization.price
    expected_quantity = optimization.expected_quantity
    expected_profit = (price - sku.base_cost) * expected_quantity
    expected_margin_pct = (price - sku.base_cost) / price if price > 0 else 0.0

    return SkuPricingResult(
        sku_id=sku.sku_id,
        recommended_price=round(price, 4),
        expected_quantity=round(expected_quantity, 4),
        expected_margin_pct=round(expected_margin_pct, 4),
        expected_profit=round(expected_profit, 4),
        elasticity=None if elasticity.is_default else round(elasticity.slope, 4),
        used_default_elasticity=elasticity.is_default,
        constraints_applied=optimization.constraints_applied,
    )


def price_skus(request: PricingRequest) -> PricingResponse:
    results = [_price_sku(sku, request.objective) for sku in request.skus]
    return PricingResponse(results=results)
