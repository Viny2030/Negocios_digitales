"""
Optimización con restricciones sobre la curva de demanda estimada (ver
elasticity.py). Restricciones de negocio: margen mínimo sobre el costo base y
variación diaria máxima respecto al precio vigente.

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

from dataclasses import dataclass

from scipy.optimize import minimize_scalar

from app.cases.dynamic_pricing.elasticity import ElasticityModel
from app.cases.dynamic_pricing.schemas import PricingObjective


@dataclass
class PriceOptimizationResult:
    price: float
    expected_quantity: float
    constraints_applied: list[str]


def _feasible_price_range(
    base_cost: float,
    current_price: float,
    min_margin_pct: float,
    max_daily_variation_pct: float,
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

    return low, high, constraints_applied


def optimize_price(
    elasticity: ElasticityModel,
    base_cost: float,
    current_price: float,
    min_margin_pct: float,
    max_daily_variation_pct: float,
    objective: PricingObjective,
) -> PriceOptimizationResult:
    low, high, constraints_applied = _feasible_price_range(
        base_cost, current_price, min_margin_pct, max_daily_variation_pct
    )

    if low == high:
        price = low
    elif objective == PricingObjective.volume:
        # Con demanda decreciente en precio (slope < 0, que es lo que fit_elasticity
        # garantiza salvo el caso default), el volumen esperado se maximiza en el
        # extremo inferior del rango factible -- no hace falta un solver para esto,
        # es la solución de esquina directa.
        price = low
    else:

        def negative_profit(p: float) -> float:
            return -(p - base_cost) * elasticity.expected_quantity(p)

        result = minimize_scalar(negative_profit, bounds=(low, high), method="bounded")
        price = float(result.x)

    expected_quantity = elasticity.expected_quantity(price)
    return PriceOptimizationResult(
        price=price, expected_quantity=expected_quantity, constraints_applied=constraints_applied
    )
