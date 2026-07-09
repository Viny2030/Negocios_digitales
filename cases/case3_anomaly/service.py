"""Lógica de negocio del Caso 3.3: clasifica falso positivo, decide la acción y (opcionalmente)
crea el ticket priorizado en la herramienta de gestión de la empresa (Jira/Opsgenie/Zendesk)."""
import os

from app.core.config import get_settings
from app.cases.case3_anomaly.classifier import FalsePositiveClassifier
from app.cases.case3_anomaly.policy import AnomalyTriagePolicy
from app.cases.case3_anomaly.schemas import AlertWebhookPayload, TeamCapacity, TriageAction, TriageDecision

settings = get_settings()

_policy_cache: dict[int, AnomalyTriagePolicy] = {}
_classifier_cache: dict[int, FalsePositiveClassifier] = {}


def _get_policy_for_company(company_id: int) -> AnomalyTriagePolicy:
    if company_id not in _policy_cache:
        model_path = os.path.join(settings.MODELS_DIR, str(company_id), "case3_anomaly", "model.zip")
        _policy_cache[company_id] = AnomalyTriagePolicy(model_path=model_path)
    return _policy_cache[company_id]


def _get_classifier_for_company(company_id: int) -> FalsePositiveClassifier:
    if company_id not in _classifier_cache:
        model_path = os.path.join(settings.MODELS_DIR, str(company_id), "case3_anomaly", "fp_classifier.joblib")
        _classifier_cache[company_id] = FalsePositiveClassifier(model_path=model_path)
    return _classifier_cache[company_id]


def _create_ticket_if_needed(action: TriageAction, payload: AlertWebhookPayload) -> bool:
    """Stub de integración con Jira/Opsgenie/Zendesk. Reemplazar por la llamada real a su REST API,
    usando las credenciales que la empresa configure (ver .env.example / tabla companies)."""
    if action in (TriageAction.investigate_light, TriageAction.investigate_deep,
                  TriageAction.escalate_engineering, TriageAction.escalate_business):
        # TODO: POST a la API de Jira/Opsgenie con payload.alert_id, prioridad, contexto
        return True
    return False


def handle_alert_webhook(
    company_id: int, payload: AlertWebhookPayload, capacity: TeamCapacity,
) -> TriageDecision:
    classifier = _get_classifier_for_company(company_id)
    fp_proba = classifier.predict_false_positive_proba(payload)

    policy = _get_policy_for_company(company_id)
    action, priority_score = policy.decide(payload, capacity, fp_proba)

    ticket_created = _create_ticket_if_needed(action, payload)

    rationale = (
        f"z-score={payload.z_score:.2f}, prob. falso positivo={fp_proba:.2f}, "
        f"capacidad disponible={capacity.analyst_hours_available}h."
    )
    return TriageDecision(
        alert_id=payload.alert_id,
        action=action,
        priority_score=priority_score,
        false_positive_probability=round(fp_proba, 3),
        rationale=rationale,
        ticket_created=ticket_created,
    )


def invalidate_policy_cache(company_id: int) -> None:
    _policy_cache.pop(company_id, None)
    _classifier_cache.pop(company_id, None)
