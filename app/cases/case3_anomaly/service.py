"""Lógica de negocio del Caso 3.3: clasifica falso positivo, decide la acción y (opcionalmente)
crea el ticket priorizado en la herramienta de gestión de la empresa (Jira/Opsgenie/Zendesk)."""

import logging

import httpx

import model_registry
from app.cases.case3_anomaly.classifier import FalsePositiveClassifier
from app.cases.case3_anomaly.policy import AnomalyTriagePolicy
from app.cases.case3_anomaly.schemas import AlertWebhookPayload, TeamCapacity, TriageAction, TriageDecision
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
CASE = "case3_anomaly"

# Ver comentario equivalente en case1_messaging/service.py.
_policy_cache: dict[int, tuple[str | None, AnomalyTriagePolicy]] = {}
_classifier_cache: dict[int, tuple[str | None, FalsePositiveClassifier]] = {}


def _get_policy_for_company(company_id: int) -> AnomalyTriagePolicy:
    current_version_id = model_registry.get_current_version_id(settings.MODELS_DIR, company_id, CASE)
    cached = _policy_cache.get(company_id)
    if cached is None or cached[0] != current_version_id:
        model_path = model_registry.get_current_artifact_path(settings.MODELS_DIR, company_id, CASE, "model.zip")
        cached = (current_version_id, AnomalyTriagePolicy(model_path=model_path))
        _policy_cache[company_id] = cached
    return cached[1]


def _get_classifier_for_company(company_id: int) -> FalsePositiveClassifier:
    current_version_id = model_registry.get_current_version_id(settings.MODELS_DIR, company_id, CASE)
    cached = _classifier_cache.get(company_id)
    if cached is None or cached[0] != current_version_id:
        model_path = model_registry.get_current_artifact_path(
            settings.MODELS_DIR,
            company_id,
            CASE,
            "fp_classifier.joblib",
        )
        cached = (current_version_id, FalsePositiveClassifier(model_path=model_path))
        _classifier_cache[company_id] = cached
    return cached[1]


def _create_ticket_if_needed(company_id: int, action: TriageAction, payload: AlertWebhookPayload) -> bool:
    """Si la acción de triaje amerita intervención humana, crea el ticket en la herramienta de
    gestión de la empresa vía un webhook HTTP genérico (Jira/Opsgenie/Zendesk automation webhook,
    o un endpoint intermedio propio que la empresa configure). La URL sale de
    settings.TICKETING_WEBHOOK_URL (ver .env.example); mientras no esté configurada, no hay
    integración real disponible: se deja constancia en el log y se sigue sin crear el ticket, en
    vez de hacer fallar el triaje completo por un problema de configuración externo."""
    necesita_ticket = action in (
        TriageAction.investigate_light,
        TriageAction.investigate_deep,
        TriageAction.escalate_engineering,
        TriageAction.escalate_business,
    )
    if not necesita_ticket:
        return False

    if not settings.TICKETING_WEBHOOK_URL:
        logger.info(
            "Ticketing no configurado (TICKETING_WEBHOOK_URL vacío); no se crea ticket para "
            "alert_id=%s (empresa=%s, acción=%s).",
            payload.alert_id,
            company_id,
            action.value,
        )
        return False

    ticket_payload = {
        "company_id": company_id,
        "alert_id": payload.alert_id,
        "action": action.value,
        "z_score": payload.z_score,
        "source": "prescriptive-platform/case3_anomaly",
    }
    try:
        response = httpx.post(
            settings.TICKETING_WEBHOOK_URL,
            json=ticket_payload,
            timeout=settings.TICKETING_WEBHOOK_TIMEOUT_S,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        logger.exception(
            "Falló la creación del ticket para alert_id=%s (empresa=%s) contra %s.",
            payload.alert_id,
            company_id,
            settings.TICKETING_WEBHOOK_URL,
        )
        return False

    logger.info(
        "Ticket creado para alert_id=%s (empresa=%s, acción=%s).", payload.alert_id, company_id, action.value
    )
    return True


def handle_alert_webhook(
    company_id: int,
    payload: AlertWebhookPayload,
    capacity: TeamCapacity,
) -> TriageDecision:
    classifier = _get_classifier_for_company(company_id)
    fp_proba = classifier.predict_false_positive_proba(payload)

    policy = _get_policy_for_company(company_id)
    action, priority_score = policy.decide(payload, capacity, fp_proba)

    ticket_created = _create_ticket_if_needed(company_id, action, payload)

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
