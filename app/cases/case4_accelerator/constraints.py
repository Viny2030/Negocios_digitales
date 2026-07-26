"""
Filtro post-RL (Caso 6.2): dado un score de valor esperado por startup (estimado por
la política, RL o heurística), resuelve la asignación óptima de los recursos del
período (mentoría, capital follow-on, conexiones estratégicas) sobre el portfolio,
sujeta a la capacidad total disponible y a un tope opcional de concentración por
startup. Mismo patrón de "filtro de factibilidad post-RL vía PuLP" que
case1_messaging/constraints.py y case2_pricing/constraints.py.

A diferencia de esos dos casos (que ajustan una única decisión para minimizar la
desviación respecto de lo recomendado), acá el LP resuelve directamente el problema
de asignación: maximizar el valor esperado total del portfolio sujeto al presupuesto
disponible. Es la forma correcta de usar un LP cuando el objetivo es lineal en las
variables de decisión (monto asignado a cada startup): el óptimo se calcula
exactamente, no se aproxima. Con el objetivo lineal y sin tope de concentración, el
óptimo naturalmente concentra el presupuesto en las startups de mayor score — el
comportamiento que el enunciado espera que el agente aprenda por sobre la
distribución uniforme.
"""
import pulp

from app.cases.case4_accelerator.schemas import ResourcePool, StartupAllocation, StartupProfile


def _allocate_continuous_resource(
    value_scores: dict[str, float], total_available: float, max_share_per_startup: float,
) -> dict[str, float]:
    """LP: maximizar sum(value_i * x_i) sujeto a sum(x_i) <= total_available y
    0 <= x_i <= max_share_per_startup * total_available (tope de concentración).
    Las variables se indexan por posición (no por startup_id) porque PuLP no acepta
    cualquier caracter en el nombre de una variable, y los IDs reales de startups
    suelen traer guiones, espacios o puntos (ej. "airbnb-2019")."""
    if total_available <= 0 or not value_scores:
        return {sid: 0.0 for sid in value_scores}

    ids = list(value_scores.keys())
    prob = pulp.LpProblem("resource_allocation", pulp.LpMaximize)
    cap = max_share_per_startup * total_available
    x = {i: pulp.LpVariable(f"x_{i}", lowBound=0, upBound=cap) for i in range(len(ids))}

    prob += pulp.lpSum(value_scores[ids[i]] * x[i] for i in range(len(ids)))
    prob += pulp.lpSum(x.values()) <= total_available

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return {ids[i]: round(float(x[i].value() or 0.0), 2) for i in range(len(ids))}


def _allocate_connections(value_scores: dict[str, float], n_available: int) -> dict[str, bool]:
    """LP de selección binaria (capacidad n_available): activar la conexión
    estratégica en las startups de mayor valor esperado, sin superar el número de
    conexiones activables del período. Mismo indexado por posición que arriba."""
    if n_available <= 0 or not value_scores:
        return {sid: False for sid in value_scores}

    ids = list(value_scores.keys())
    prob = pulp.LpProblem("connection_selection", pulp.LpMaximize)
    y = {i: pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(len(ids))}

    prob += pulp.lpSum(value_scores[ids[i]] * y[i] for i in range(len(ids)))
    prob += pulp.lpSum(y.values()) <= n_available

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return {ids[i]: bool(round(y[i].value() or 0)) for i in range(len(ids))}


def apply_portfolio_allocation(
    value_scores: dict[str, float],
    startups: list[StartupProfile],
    resources: ResourcePool,
) -> tuple[list[StartupAllocation], list[str]]:
    """Corre los 3 LPs (mentoría, capital, conexiones) y arma la lista de
    asignaciones final, una por startup del portfolio, junto con las restricciones
    que se aplicaron (para trazabilidad en la respuesta del endpoint)."""
    applied: list[str] = ["lp_asignacion_presupuesto_mentoria"]

    mentorship = _allocate_continuous_resource(
        value_scores, resources.mentorship_hours_available, resources.max_mentorship_share_per_startup,
    )
    if resources.max_mentorship_share_per_startup < 1.0:
        applied.append("tope_concentracion_mentoria")

    capital = {sid: 0.0 for sid in value_scores}
    if resources.followon_capital_available > 0:
        capital = _allocate_continuous_resource(
            value_scores, resources.followon_capital_available, resources.max_capital_share_per_startup,
        )
        applied.append("lp_asignacion_presupuesto_capital")
        if resources.max_capital_share_per_startup < 1.0:
            applied.append("tope_concentracion_capital")

    connections = _allocate_connections(value_scores, resources.strategic_connections_available)
    if resources.strategic_connections_available > 0:
        applied.append("lp_asignacion_conexiones_estrategicas")

    allocations = [
        StartupAllocation(
            startup_id=s.startup_id,
            mentorship_hours=mentorship.get(s.startup_id, 0.0),
            followon_capital_recommended=capital.get(s.startup_id, 0.0),
            strategic_connection_activated=connections.get(s.startup_id, False),
            expected_value_score=round(value_scores.get(s.startup_id, 0.0), 3),
        )
        for s in startups
    ]
    return allocations, applied