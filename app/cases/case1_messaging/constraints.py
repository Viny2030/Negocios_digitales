"""
Filtro post-RL (Caso 1.1): garantiza que el touchpoint propuesto por la política
(RL o heurística) respete las restricciones de negocio de cadencia de contacto y
presupuesto semanal por canal, ANTES de recomendarlo.

Regla de negocio por defecto: máximo 1 email (cualquier variante A/B/C) cada 7 días
por prospecto, más un tope semanal configurable para el resto de los canales.
Resuelve un LP de selección binaria (PuLP) entre los touchpoints factibles: exactamente
el mismo patrón que app/cases/case2_pricing/constraints.py, pero acá el LP elige una
acción discreta en vez de un precio continuo.
"""

import pulp

from app.cases.case1_messaging.schemas import ChannelBudget, ProspectProfile, Touchpoint

# A qué "bucket" de presupuesto pertenece cada touchpoint. Las 3 variantes de email
# comparten el mismo bucket porque la restricción de negocio es "1 email por semana",
# sin importar la variante A/B/C.
_CHANNEL_GROUP: dict[Touchpoint, str | None] = {
    Touchpoint.email_a: "email",
    Touchpoint.email_b: "email",
    Touchpoint.email_c: "email",
    Touchpoint.retargeting_meta: "retargeting_meta",
    Touchpoint.sms: "sms",
    Touchpoint.push: "push",
    Touchpoint.call: "call",
    Touchpoint.wait: None,  # esperar nunca consume presupuesto ni se bloquea
}

# Orden de repliegue cuando el touchpoint recomendado no es factible: se prueba el
# siguiente canal "razonable" antes de resignarse a esperar. No reemplaza al valor
# estimado por el modelo (policy.py no expone el ranking completo de acciones), pero
# evita que una restricción de negocio termine siempre en "wait".
_FALLBACK_ORDER = [
    Touchpoint.retargeting_meta,
    Touchpoint.sms,
    Touchpoint.push,
    Touchpoint.call,
    Touchpoint.email_a,
    Touchpoint.email_b,
    Touchpoint.email_c,
    Touchpoint.wait,
]

MIN_DAYS_BETWEEN_EMAILS = 7  # "1 mail por semana"


def _is_feasible(touchpoint: Touchpoint, profile: ProspectProfile, budget: ChannelBudget) -> bool:
    if touchpoint == Touchpoint.wait:
        return True

    group = _CHANNEL_GROUP[touchpoint]

    # Restricción de cadencia: ningún email (de ninguna variante) si el último email
    # de cualquier variante se mandó hace menos de MIN_DAYS_BETWEEN_EMAILS días.
    if group == "email" and profile.days_since_last_email is not None:
        if profile.days_since_last_email < MIN_DAYS_BETWEEN_EMAILS:
            return False

    # Restricción de presupuesto semanal por canal (contactos ya realizados esta
    # semana + este touchpoint no debe superar el tope configurado por la empresa).
    used = profile.contacts_this_week.get(group, 0)
    cap = getattr(budget, group)
    if used >= cap:
        return False

    return True


def _blocking_reason(touchpoint: Touchpoint, profile: ProspectProfile, budget: ChannelBudget) -> str:
    group = _CHANNEL_GROUP[touchpoint]
    if (
        group == "email"
        and profile.days_since_last_email is not None
        and profile.days_since_last_email < MIN_DAYS_BETWEEN_EMAILS
    ):
        return "cadencia_1_email_por_semana"
    return f"presupuesto_semanal_{group}_agotado"


def apply_frequency_and_budget_filter(
    recommended: Touchpoint,
    profile: ProspectProfile,
    budget: ChannelBudget,
) -> tuple[Touchpoint, list[str]]:
    """
    Devuelve el touchpoint final (recommended si es factible, o la mejor alternativa
    factible resuelta vía LP) junto con la lista de restricciones que se aplicaron.
    """
    applied: list[str] = []
    all_touchpoints = list(Touchpoint)
    feasible = {tp: _is_feasible(tp, profile, budget) for tp in all_touchpoints}

    if feasible[recommended]:
        return recommended, applied

    applied.append(_blocking_reason(recommended, profile, budget))

    # LP de selección binaria: entre los touchpoints factibles, elegir el de menor
    # costo de preferencia (equivalente a "el más parecido a lo recomendado que
    # todavía se pueda hacer"). Con un solo touchpoint por decisión el problema es
    # trivial de resolver a mano, pero se deja como LP porque el enunciado del caso
    # pide explícitamente un módulo LP para estas restricciones, y este mismo molde
    # escala sin cambios si en el futuro se agregan más restricciones simultáneas
    # (ej. presupuesto mensual agregado, tope por par empresa-canal, etc.).
    prob = pulp.LpProblem("touchpoint_feasibility", pulp.LpMinimize)
    x = {tp: pulp.LpVariable(f"x_{tp.value}", cat="Binary") for tp in all_touchpoints}

    prob += pulp.lpSum(x.values()) == 1
    for tp in all_touchpoints:
        if not feasible[tp]:
            prob += x[tp] == 0

    preference_cost = {tp: rank for rank, tp in enumerate(_FALLBACK_ORDER)}
    prob += pulp.lpSum(preference_cost[tp] * x[tp] for tp in all_touchpoints)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    chosen = next(tp for tp in all_touchpoints if x[tp].value() == 1)
    applied.append("lp_seleccion_alternativa_factible")
    return chosen, applied
