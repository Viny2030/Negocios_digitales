"""
Filtro post-RL (Caso 2.1): garantiza que el precio propuesto por el agente sea factible
antes de mostrarlo al usuario. Combina:
  1) Programación Lineal simple (bandas de precio, coherencia entre planes) vía PuLP.
  2) Métricas de equidad (Demographic Parity) vía Fairlearn, usadas en evaluación/reentrenamiento
     más que en el filtro sincrónico (que debe responder en <50ms).
"""
import pulp

from app.cases.case2_pricing.schemas import PriceRequest


def apply_feasibility_filter(raw_price: float, request: PriceRequest) -> tuple[float, list[str]]:
    """Resuelve un LP mínimo: precio más cercano al propuesto por el agente que respete
    piso de margen, techo de banda y coherencia con el precio de competencia."""
    applied: list[str] = []

    floor_price = request.base_cost * (1 + request.min_margin_pct)
    ceiling_price = request.base_cost * request.max_multiplier

    prob = pulp.LpProblem("price_feasibility", pulp.LpMinimize)
    price_var = pulp.LpVariable("price", lowBound=floor_price, upBound=ceiling_price)
    deviation = pulp.LpVariable("deviation", lowBound=0)

    prob += deviation
    prob += deviation >= price_var - raw_price
    prob += deviation >= raw_price - price_var

    # Coherencia competitiva: no superar en más de 20% el precio promedio de mercado
    if request.competitor_avg_price:
        competitive_ceiling = request.competitor_avg_price * 1.2
        if competitive_ceiling < ceiling_price:
            prob += price_var <= competitive_ceiling
            applied.append("coherencia_competitiva")

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    final_price = float(pulp.value(price_var))

    if abs(final_price - raw_price) > 1e-6:
        applied.append("ajuste_a_banda_permitida")
    if final_price <= floor_price + 1e-6:
        applied.append("margen_minimo_aplicado")

    return round(final_price, 2), applied
