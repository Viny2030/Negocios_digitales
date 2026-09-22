"""
Asignación ponderada lead-vendedor vía algoritmo húngaro
(scipy.optimize.linear_sum_assignment), maximizando la suma de
`conversion_probability * afinidad_del_rep` sobre todas las asignaciones.

Para que el húngaro (que asigna 1 a 1) respete la capacidad de cada vendedor
(puede tomar más de un lead por ronda), cada rep se expande en `capacity`
columnas idénticas de la matriz de scores -- así el solver puede asignarle
hasta `capacity` leads sin tratamiento especial. scipy soporta matrices
rectangulares de forma nativa (más leads que lugares, o al revés), así que no
hace falta rellenar con columnas/filas dummy: si sobran leads sin ningún slot
libre, quedan simplemente fuera de la asignación devuelta.
"""

from scipy.optimize import linear_sum_assignment

from app.cases.lead_match.schemas import LeadAssignment, LeadFeatures, SalesRep


def _affinity(rep: SalesRep, lead: LeadFeatures, specialty_bonus: float) -> float:
    bonus = specialty_bonus if lead.origin in rep.specialties else 1.0
    return rep.historical_close_rate * bonus


def match_leads(
    leads: list[LeadFeatures],
    conversion_probabilities: dict[str, float],
    reps: list[SalesRep],
    specialty_bonus: float,
) -> tuple[list[LeadAssignment], list[str]]:
    capacitated_reps = [rep for rep in reps if rep.capacity > 0]
    if not leads or not capacitated_reps:
        return [], [lead.lead_id for lead in leads]

    # slot_rep_ids[j] = a qué rep_id corresponde la columna j de la matriz.
    slot_rep_ids: list[str] = []
    for rep in capacitated_reps:
        slot_rep_ids.extend([rep.rep_id] * rep.capacity)

    reps_by_id = {rep.rep_id: rep for rep in capacitated_reps}
    score_matrix = [
        [
            conversion_probabilities[lead.lead_id] * _affinity(reps_by_id[rep_id], lead, specialty_bonus)
            for rep_id in slot_rep_ids
        ]
        for lead in leads
    ]

    # linear_sum_assignment MINIMIZA costo -- para maximizar el score total se
    # le pasa el negativo de la matriz.
    cost_matrix = [[-score for score in row] for row in score_matrix]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assigned_lead_indices = set(row_ind.tolist())
    assignments = [
        LeadAssignment(
            lead_id=leads[i].lead_id,
            rep_id=slot_rep_ids[j],
            conversion_probability=round(conversion_probabilities[leads[i].lead_id], 4),
            match_score=round(score_matrix[i][j], 4),
        )
        for i, j in zip(row_ind.tolist(), col_ind.tolist(), strict=True)
    ]
    unmatched = [lead.lead_id for idx, lead in enumerate(leads) if idx not in assigned_lead_indices]

    return assignments, unmatched
