import datetime as dt

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier

from app.cases.case3_anomaly.classifier import DEFAULT_FEATURE_COLS, FalsePositiveClassifier
from app.cases.case3_anomaly.schemas import AlertWebhookPayload


def _payload(z_score=3.5, pct_change=-12.5, historical_similar_count=1, estimated_financial_impact=8000.0):
    return AlertWebhookPayload(
        alert_id="alert-features-1",
        metric_name="checkout_conversion",
        z_score=z_score,
        pct_change=pct_change,
        triggered_at=dt.datetime(2026, 7, 9, 10, 0, 0),
        historical_similar_count=historical_similar_count,
        estimated_financial_impact=estimated_financial_impact,
    )


def test_alert_webhook_returns_decision(client, company_api_key):
    payload = {
        "alert_id": "alert-1",
        "metric_name": "checkout_conversion",
        "z_score": 3.5,
        "pct_change": -12.5,
        "triggered_at": "2026-07-09T10:00:00",
        "business_context": "checkout",
        "historical_similar_count": 1,
        "estimated_financial_impact": 8000,
    }
    resp = client.post("/api/v1/case3/alert-webhook", json=payload, headers={"X-API-Key": company_api_key})
    assert resp.status_code == 200
    body = resp.json()
    assert body["alert_id"] == "alert-1"
    assert body["action"] in [
        "ignore",
        "investigate_light",
        "investigate_deep",
        "escalate_engineering",
        "escalate_business",
        "merge_related",
    ]


def _fit_single_class_classifier(tmp_path, label_value: int):
    """Entrena un RandomForestClassifier real con `is_false_positive` (o su feature vector
    equivalente) todo un mismo valor -- realista para una empresa nueva con pocos tickets
    históricos, donde `model.classes_` termina con longitud 1."""
    X = [[3.0, -10.0, 1, 5000.0], [1.5, 5.0, 0, 0.0], [4.0, -2.0, 2, 12000.0]]
    y = [label_value, label_value, label_value]
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    assert len(clf.classes_) == 1  # confirma que este es justo el caso límite del bug

    model_path = tmp_path / "fp_classifier.joblib"
    joblib.dump(clf, model_path)
    return str(model_path)


@pytest.mark.parametrize("label_value", [0, 1])
def test_predict_proba_no_explota_con_una_sola_clase(tmp_path, label_value):
    """Bug 1: si el historial de la empresa sólo tiene una clase (is_false_positive todo 0 o
    todo 1), `model.classes_` tiene longitud 1 y `proba[1]` explotaba con IndexError."""
    model_path = _fit_single_class_classifier(tmp_path, label_value)
    classifier = FalsePositiveClassifier(model_path=model_path)

    proba = classifier.predict_false_positive_proba(_payload())

    assert 0.0 <= proba <= 1.0
    if label_value == 1:
        # La única clase observada es "false_positive": la probabilidad debe ser máxima (1.0),
        # no 0.0 -- si el bug se "arreglara" mal (siempre devolver 0.0 en vez de mapear por
        # clase), este assert lo detecta.
        assert proba == pytest.approx(1.0)
    else:
        # La única clase observada es "real_incident": probabilidad de false_positive es 0.0.
        assert proba == pytest.approx(0.0)


def test_payload_to_features_usa_z_score_con_signo_y_sin_binarizar_impacto():
    """Bug 2: entrenamiento (scripts/train_case3.py::_train_fp_classifier) usa z_score con signo
    y el valor crudo de estimated_financial_impact -- _payload_to_features no debe usar abs() ni
    binarizar, o el modelo recibe en inferencia features que no tienen nada que ver con lo que
    aprendió en entrenamiento."""
    classifier = FalsePositiveClassifier(model_path=None)  # sin modelo, para inspeccionar features

    payload = _payload(z_score=-4.2, estimated_financial_impact=15000.0)
    features = classifier._payload_to_features(payload)

    assert classifier.feature_cols == DEFAULT_FEATURE_COLS
    row = dict(zip(classifier.feature_cols, features[0], strict=True))

    # Un z_score negativo NO debe convertirse en positivo (eso es lo que hacía abs()).
    assert row["z_score"] == pytest.approx(-4.2)
    # El impacto financiero debe viajar en su magnitud real, no binarizado a 0.0/1.0.
    assert row["estimated_financial_impact"] == pytest.approx(15000.0)


def test_payload_to_features_respeta_columnas_persistidas_en_metadata(tmp_path):
    """Bug 3: si el dataset de entrenamiento no traía alguna columna (ver `available_cols` en
    scripts/train_case3.py), el modelo se entrenó con menos features. El vector de inferencia
    debe construirse con exactamente esas columnas (y ese orden), no con la lista completa por
    defecto, o `model.predict_proba` explota con un mismatch de shape."""
    reduced_cols = ["z_score", "historical_similar_count"]
    X = [[3.0, 1], [1.5, 0], [4.0, 2], [0.5, 5]]
    y = [1, 0, 1, 0]
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    model_path = tmp_path / "fp_classifier_reduced.joblib"
    joblib.dump(clf, model_path)

    classifier = FalsePositiveClassifier(model_path=str(model_path), feature_cols=reduced_cols)
    features = classifier._payload_to_features(_payload(z_score=-4.2, historical_similar_count=3))

    assert features.shape == (1, 2)
    assert list(features[0]) == [-4.2, 3]

    # No debe explotar por mismatch de shape (2 features esperadas, 2 provistas).
    proba = classifier.predict_false_positive_proba(_payload(z_score=-4.2, historical_similar_count=3))
    assert 0.0 <= proba <= 1.0
