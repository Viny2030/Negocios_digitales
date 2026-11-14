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
        stock=sku.stock,
        competitor_price=sku.competitor_price,
        max_competitor_premium_pct=sku.max_competitor_premium_pct,
    )

    price = optimization.price
    quantity = optimization.expected_quantity
    unit_profit = price - sku.base_cost

    return SkuPricingResult(
        sku_id=sku.sku_id,
        recommended_price=round(price, 4),
        expected_quantity=None if quantity is None else round(quantity, 4),
        expected_margin_pct=round(unit_profit / price, 4),
        expected_margin_on_cost_pct=round(unit_profit / sku.base_cost, 4),
        expected_profit=None if quantity is None else round(unit_profit * quantity, 4),
        elasticity=None if elasticity.is_default else round(elasticity.slope, 4),
        used_default_elasticity=elasticity.is_default,
        constraints_applied=optimization.constraints_applied,
    )


def price_skus(request: PricingRequest) -> PricingResponse:
    results = [_price_sku(sku, request.objective) for sku in request.skus]
    return PricingResponse(results=results)
