"""
Estimación de elasticidad precio-demanda por regresión log-log:

    log(quantity) = a + b * log(price)      =>      quantity(price) = exp(a) * price**b

`b` es la elasticidad-precio de la demanda (típicamente negativa: a mayor precio,
menor cantidad vendida). Se ajusta por mínimos cuadrados sobre los
`purchase_events` que manda el cliente en cada request — ver el docstring de
schemas.py sobre por qué esto es on-the-fly y no un modelo entrenado aparte.

Con pocas observaciones, o si todas fueron al mismo precio (no hay variación
para estimar una pendiente), no hay forma estadísticamente válida de ajustar
una recta -- se cae a DEFAULT_ELASTICITY, un valor conservador de referencia
(demanda moderadamente elástica) en vez de fallar el request. Es el mismo
patrón de "valor de respaldo" que usa LeadMatch cuando la empresa todavía no
entrenó un modelo (ver app/cases/lead_match/scoring.py).
"""

import math
from dataclasses import dataclass

MIN_OBSERVATIONS_FOR_FIT = 4
DEFAULT_ELASTICITY = -1.2
DEFAULT_LOG_INTERCEPT_QUANTITY_AT_PRICE_1 = 3.0  # log(exp(3)) ~= 20 unidades vendidas a price=1 como ancla


@dataclass
class ElasticityModel:
    intercept: float  # a
    slope: float  # b (elasticidad)
    is_default: bool

    def expected_quantity(self, price: float) -> float:
        if price <= 0:
            return 0.0
        log_q = self.intercept + self.slope * math.log(price)
        # Recorte a un piso razonable: el modelo log-log nunca toca 0 exactamente,
        # pero para precios muy altos puede predecir fracciones de unidad -- eso
        # está bien para el objetivo de optimización (es una expectativa, no un
        # conteo), no hace falta redondear acá.
        return max(math.exp(log_q), 0.0)


def fit_elasticity(events: list[tuple[float, int]]) -> ElasticityModel:
    """`events` es una lista de (price, quantity_sold). Devuelve el modelo ajustado,
    o el de default si no hay suficiente señal (ver MIN_OBSERVATIONS_FOR_FIT)."""
    # Solo sirven observaciones con cantidad > 0: log(0) no está definido, y una
    # venta en cero unidades a un precio dado no aporta información sobre la
    # pendiente de la curva de demanda dentro de este ajuste log-log.
    usable = [(p, q) for p, q in events if p > 0 and q > 0]

    prices = {p for p, _ in usable}
    if len(usable) < MIN_OBSERVATIONS_FOR_FIT or len(prices) < 2:
        return ElasticityModel(
            intercept=DEFAULT_LOG_INTERCEPT_QUANTITY_AT_PRICE_1,
            slope=DEFAULT_ELASTICITY,
            is_default=True,
        )

    log_p = [math.log(p) for p, _ in usable]
    log_q = [math.log(q) for _, q in usable]
    n = len(usable)
    mean_x = sum(log_p) / n
    mean_y = sum(log_q) / n
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_p, log_q, strict=True))
    ss_xx = sum((x - mean_x) ** 2 for x in log_p)

    if ss_xx == 0:
        # No debería pasar (ya filtramos >= 2 precios distintos), pero por las dudas
        # ante colinealidad numérica no dividimos por cero.
        return ElasticityModel(
            intercept=DEFAULT_LOG_INTERCEPT_QUANTITY_AT_PRICE_1,
            slope=DEFAULT_ELASTICITY,
            is_default=True,
        )

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x

    if slope >= 0:
        # Pendiente positiva (a más precio, más venta) es económicamente rara y casi
        # siempre síntoma de pocos datos ruidosos o de un efecto de demanda que no es
        # precio (ej. un producto que subió de precio Y de popularidad a la vez). No
        # hay forma de optimizar un precio de forma sensata sobre esa curva (el
        # "óptimo" sería subir el precio sin límite), así que se cae al default en
        # vez de servir una recomendación económicamente inconsistente.
        return ElasticityModel(
            intercept=DEFAULT_LOG_INTERCEPT_QUANTITY_AT_PRICE_1,
            slope=DEFAULT_ELASTICITY,
            is_default=True,
        )

    return ElasticityModel(intercept=intercept, slope=slope, is_default=False)
