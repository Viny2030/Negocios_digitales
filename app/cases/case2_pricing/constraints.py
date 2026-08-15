"""
Filtro post-RL (Caso 2.1): garantiza que el precio propuesto por el agente sea factible
antes de mostrarlo al usuario. Combina:
  1) Ajuste a la banda factible (piso de margen, techo de banda, coherencia competitiva).
  2) Métricas de equidad (Demographic Parity) vía Fairlearn, usadas en evaluación/reentrenamiento
     más que en el filtro sincrónico (que debe responder en <50ms).

Nota sobre (1): antes esto se resolvía con un LP (PuLP + CBC) por request, igual que en
case1/case4. Ahí sí hace falta un solver: hay múltiples variables compitiendo por un
recurso compartido (selección de touchpoint, asignación de portfolio). Acá el problema
tiene una sola variable (el precio) con cota inferior, cota superior y a lo sumo una
restricción lineal extra (techo competitivo) -- el óptimo de "minimizar la distancia al
precio propuesto sujeto a floor <= precio <= techo" es, por construcción, el clamp del
valor propuesto a ese rango. Resolverlo en Python puro evita lanzar el proceso del
solver CBC en el hot path del endpoint, que es el que tiene el objetivo real de <50ms.
"""

from app.cases.case2_pricing.schemas import PriceRequest


def apply_feasibility_filter(raw_price: float, request: PriceRequest) -> tuple[float, list[str]]:
    """Ajusta el precio propuesto por el agente al valor factible más cercano dentro de
    [piso_de_margen, techo], donde el techo es el menor entre la banda máxima permitida
    y (si es más estricto) el techo de coherencia competitiva."""
    applied: list[str] = []

    floor_price = request.base_cost * (1 + request.min_margin_pct)
    ceiling_price = request.base_cost * request.max_multiplier

    # Coherencia competitiva: no superar en más de 20% el precio promedio de mercado
    effective_ceiling = ceiling_price
    if request.competitor_avg_price:
        competitive_ceiling = request.competitor_avg_price * 1.2
        if competitive_ceiling < effective_ceiling:
            effective_ceiling = competitive_ceiling
            applied.append("coherencia_competitiva")

    if floor_price > effective_ceiling:
        # Caso borde que el LP original tampoco resolvía de forma explícita (habría
        # quedado infactible): el margen mínimo exigido por sí solo ya supera el techo
        # permitido. Priorizamos el margen mínimo -- regla de negocio más dura que la
        # coherencia competitiva -- y dejamos constancia explícita de la excepción.
        final_price = floor_price
        applied.append("margen_minimo_prioritario_sobre_techo")
    else:
        final_price = min(max(raw_price, floor_price), effective_ceiling)

    if abs(final_price - raw_price) > 1e-6:
        applied.append("ajuste_a_banda_permitida")
    if final_price <= floor_price + 1e-6 and "margen_minimo_prioritario_sobre_techo" not in applied:
        applied.append("margen_minimo_aplicado")

    return round(final_price, 2), applied
