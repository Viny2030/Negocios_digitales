"""
Optimización con restricciones sobre la curva de demanda estimada (ver
elasticity.py). Restricciones de negocio: margen mínimo sobre el costo base,
variación diaria máxima respecto al precio vigente y, si se informa, un techo
respecto del precio del competidor. El stock limita las unidades vendibles.

Sin datos para estimar la demanda, o sin stock, no hay fundamento para mover el
precio: se mantiene el actual y solo se ajusta si rompe una de esas reglas.

La curva de beneficio profit(price) = (price - cost) * quantity(price), con
quantity(price) = exp(a) * price**b (log-log), es no lineal -- por eso el
"programación lineal o cuadrática" de la propuesta se resuelve acá con un
optimizador escalar acotado (scipy.optimize.minimize_scalar, method="bounded")
en vez de PuLP: es la generalización directa del mismo enfoque (optimizar una
función objetivo dentro de una región factible definida por las restricciones)
al caso en que la función objetivo no es lineal. El feasible set en sí --
[precio_minimo_por_margen, precio_maximo_por_variacion] ∩ [precio_minimo_por_
variacion, precio_maximo_por_variacion] -- sigue siendo un intervalo simple,
así que no hace falta un solver LP/QP de propósito general para esto.
"""

import math
from dataclasses import dataclass

from scipy.optimize import minimize_scalar

from app.cases.dynamic_pricing.elasticity import ElasticityModel
from app.cases.dynamic_pricing.schemas import PricingObjective


@dataclass
class PriceOptimizationResult:
    price: float
    expected_quantity: float | None  # None: no hubo datos para estimar la demanda
    constraints_applied: list[str]


def _feasible_price_range(
    base_cost: float,
    current_price: float,
    min_margin_pct: float,
    max_daily_variation_pct: float,
    competitor_price: float | None = None,
    max_competitor_premium_pct: float = 0.10,
) -> tuple[float, float, list[str]]:
    constraints_applied: list[str] = []

    margin_floor = base_cost * (1 + min_margin_pct)
    constraints_applied.append(f"margen_minimo>={min_margin_pct:.0%} => precio>={margin_floor:.4f}")

    variation_low = current_price * (1 - max_daily_variation_pct)
    variation_high = current_price * (1 + max_daily_variation_pct)
    constraints_applied.append(
        f"variacion_diaria_maxima<={max_daily_variation_pct:.0%} => "
        f"precio en [{variation_low:.4f}, {variation_high:.4f}]"
    )

    low = max(margin_floor, variation_low)
    high = variation_high

    if low > high:
        # Las dos restricciones son incompatibles (el piso de margen exige subir el
        # precio más de lo que la variación diaria máxima permite hoy). En vez de
        # fallar el request, se prioriza el margen mínimo (es una regla de negocio
        # dura -- vender bajo ese margen no es una opción válida aunque implique
        # superar el límite de variación diaria por esta vez) y se deja constancia
        # explícita en constraints_applied para que quede claro en la respuesta.
        constraints_applied.append(
            "conflicto: el piso de margen mínimo excede el techo de variación diaria "
            f"máxima -- se prioriza el margen mínimo (precio={margin_floor:.4f})"
        )
        return margin_floor, margin_floor, constraints_applied

    if competitor_price is not None:
        competitor_cap = competitor_price * (1 + max_competitor_premium_pct)
        if competitor_cap >= high:
            constraints_applied.append(f"competidor: el techo de {competitor_cap:.4f} no limita el rango")
        elif competitor_cap >= low:
            high = competitor_cap
            constraints_applied.append(f"competidor: precio<={competitor_cap:.4f}")
        else:
            high = low
            constraints_applied.append(
                f"conflicto: el techo por competidor ({competitor_cap:.4f}) queda por debajo del "
                f"mínimo permitido por margen y variación diaria -- se usa ese mínimo ({low:.4f})"
            )

    return low, high, constraints_applied


def optimize_price(
    elasticity: ElasticityModel,
    base_cost: float,
    current_price: float,
    min_margin_pct: float,
    max_daily_variation_pct: float,
    objective: PricingObjective,
    stock: int | None = None,
    competitor_price: float | None = None,
    max_competitor_premium_pct: float = 0.10,
) -> PriceOptimizationResult:
    low, high, constraints_applied = _feasible_price_range(
        base_cost,
        current_price,
        min_margin_pct,
        max_daily_variation_pct,
        competitor_price,
        max_competitor_premium_pct,
    )
    current_within_rules = min(max(current_price, low), high)

    if elasticity.is_default:
        constraints_applied.append(
            "sin_datos: no se pudo estimar la demanda -- se mantiene el precio actual, "
            "ajustado solo si rompe una regla"
        )
        return PriceOptimizationResult(current_within_rules, None, constraints_applied)

    if stock == 0:
        constraints_applied.append("sin_stock: no hay unidades para vender -- se mantiene el precio actual")
        return PriceOptimizationResult(current_within_rules, 0.0, constraints_applied)

    def sellable_quantity(p: float) -> float:
        demand = elasticity.expected_quantity(p)
        return demand if stock is None else min(demand, float(stock))

    if low == high:
        price = low
    elif objective == PricingObjective.volume:
        # La demanda baja con el precio, así que el mayor volumen está en el piso del
        # rango. Con stock limitado, alcanza con el precio al que la demanda iguala al
        # stock: bajar más no vende una unidad extra y solo resigna ganancia.
        price = low
        if stock is not None:
            price_selling_all_stock = math.exp((math.log(stock) - elasticity.intercept) / elasticity.slope)
            price = min(max(price_selling_all_stock, low), high)
    else:

        def profit(p: float) -> float:
            return (p - base_cost) * sellable_quantity(p)

        result = minimize_scalar(lambda p: -profit(p), bounds=(low, high), method="bounded")
        # El método acotado nunca evalúa los extremos exactos, y el óptimo puede estar en uno.
        price = max(float(result.x), low, high, key=profit)

    return PriceOptimizationResult(price, sellable_quantity(price), constraints_applied)
