"""Tests de LeadMatch Engine: matching húngaro con capacidad, heurística de
respaldo, entrenamiento del scorer y el flujo end-to-end."""

import io

import pandas as pd
import pytest

from app.cases.lead_match.matching import match_leads
from app.cases.lead_match.schemas import LeadFeatures, SalesRep
from app.cases.lead_match.scoring import ConversionScorer
from app.cases.lead_match.train import InsufficientTrainingDataError, train_for_company


def _lead(lead_id, origin="organic", time_on_site=10.0, company_size=50) -> LeadFeatures:
    return LeadFeatures(
        lead_id=lead_id, origin=origin, time_on_site_minutes=time_on_site, company_size=company_size
    )


def test_heuristica_devuelve_probabilidad_entre_0_y_1():
    scorer = ConversionScorer(model=None, is_default=True)
    p_bajo = scorer.predict_proba(_lead("l1", time_on_site=0, company_size=0))
    p_alto = scorer.predict_proba(_lead("l2", time_on_site=60, company_size=1000))
    assert 0.0 <= p_bajo <= 1.0
    assert 0.0 <= p_alto <= 1.0
    assert p_alto > p_bajo  # más tiempo en sitio + empresa más grande => mayor score heurístico


def test_match_respeta_capacidad_del_rep():
    leads = [_lead(f"l{i}") for i in range(5)]
    reps = [SalesRep(rep_id="r1", capacity=2), SalesRep(rep_id="r2", capacity=3)]
    probs = {lead.lead_id: 0.5 for lead in leads}

    assignments, unmatched = match_leads(leads, probs, reps, specialty_bonus=1.2)

    assert len(assignments) == 5
    assert unmatched == []
    counts: dict[str, int] = {}
    for a in assignments:
        counts[a.rep_id] = counts.get(a.rep_id, 0) + 1
    assert counts["r1"] <= 2
    assert counts["r2"] <= 3


def test_match_deja_leads_sin_asignar_si_falta_capacidad():
    leads = [_lead(f"l{i}") for i in range(5)]
    reps = [SalesRep(rep_id="r1", capacity=2)]
    probs = {lead.lead_id: 0.5 for lead in leads}

    assignments, unmatched = match_leads(leads, probs, reps, specialty_bonus=1.2)

    assert len(assignments) == 2
    assert len(unmatched) == 3


def test_match_prioriza_reps_con_mejor_afinidad():
    # Un solo lead, dos reps con la misma tasa de cierre pero uno especializado
    # en el origen del lead -- debe ganarle al genérico por el bonus.
    lead = _lead("l1", origin="paid_ads")
    reps = [
        SalesRep(rep_id="generico", capacity=1, historical_close_rate=0.5),
        SalesRep(rep_id="especialista", capacity=1, historical_close_rate=0.5, specialties=["paid_ads"]),
    ]
    probs = {"l1": 0.8}

    assignments, unmatched = match_leads([lead], probs, reps, specialty_bonus=1.5)

    assert unmatched == []
    assert assignments[0].rep_id == "especialista"


def test_train_falla_con_dataset_insuficiente(tmp_path):
    csv_path = tmp_path / "chico.csv"
    pd.DataFrame(
        {
            "lead_id": ["a", "b"],
            "origin": ["organic", "paid_ads"],
            "time_on_site_minutes": [5, 10],
            "company_size": [10, 20],
            "rep_id": ["r1", "r2"],
            "converted": [1, 0],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(InsufficientTrainingDataError):
        train_for_company(company_id=999, dataset_path=str(csv_path))


def test_train_y_promote_deja_modelo_servible(tmp_path, monkeypatch):
    import numpy as np

    from config import get_settings

    models_dir = tmp_path / "models"
    monkeypatch.setattr(get_settings(), "MODELS_DIR", str(models_dir))

    rng = np.random.default_rng(42)
    n = 200
    origins = rng.choice(["organic", "paid_ads", "referral"], n)
    time_on_site = rng.uniform(0, 60, n)
    company_size = rng.integers(1, 1000, n)
    # converted correlaciona con time_on_site para que el modelo tenga algo real que aprender.
    converted = (time_on_site > 30).astype(int)

    csv_path = tmp_path / "leads.csv"
    pd.DataFrame(
        {
            "lead_id": [f"l{i}" for i in range(n)],
            "origin": origins,
            "time_on_site_minutes": time_on_site,
            "company_size": company_size,
            "rep_id": rng.choice(["r1", "r2", "r3"], n),
            "converted": converted,
        }
    ).to_csv(csv_path, index=False)

    version_id, metrics = train_for_company(company_id=123, dataset_path=str(csv_path))

    assert version_id
    assert metrics["n_train"] > 0
    import model_registry

    artifact = model_registry.get_current_artifact_path(str(models_dir), 123, "lead_match", "model.joblib")
    assert artifact is not None


def test_endpoint_match_usa_heuristica_sin_modelo_entrenado(client, company_api_key):
    payload = {
        "leads": [{"lead_id": "l1", "origin": "organic", "time_on_site_minutes": 15, "company_size": 100}],
        "reps": [{"rep_id": "r1", "capacity": 1}],
    }
    resp = client.post("/api/v1/lead-match/match", json=payload, headers={"X-API-Key": company_api_key})
    assert resp.status_code == 200
    body = resp.json()
    assert body["used_default_scoring"] is True
    assert body["assignments"][0]["rep_id"] == "r1"


def test_endpoint_match_requiere_api_key(client):
    payload = {
        "leads": [{"lead_id": "l1", "origin": "organic", "time_on_site_minutes": 15, "company_size": 100}],
        "reps": [{"rep_id": "r1", "capacity": 1}],
    }
    resp = client.post("/api/v1/lead-match/match", json=payload)
    assert resp.status_code == 401


def test_endpoint_train_end_to_end(client, company_api_key):
    import numpy as np

    rng = np.random.default_rng(7)
    n = 60
    time_on_site = rng.uniform(0, 60, n)
    converted = (time_on_site > 30).astype(int)
    header = "lead_id,origin,time_on_site_minutes,company_size,rep_id,converted\n"
    rows = "".join(
        f"l{i},organic,{time_on_site[i]},{int(rng.integers(1, 500))},r1,{converted[i]}\n" for i in range(n)
    )
    csv_bytes = (header + rows).encode()

    upload = client.post(
        "/api/v1/datasets/lead_match",
        files={"file": ("leads.csv", io.BytesIO(csv_bytes), "text/csv")},
        headers={"X-API-Key": company_api_key},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["id"]

    resp = client.post(
        "/api/v1/lead-match/train",
        json={"dataset_id": dataset_id},
        headers={"X-API-Key": company_api_key},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "trained"

    match_resp = client.post(
        "/api/v1/lead-match/match",
        json={
            "leads": [{"lead_id": "l1", "origin": "organic", "time_on_site_minutes": 45, "company_size": 50}],
            "reps": [{"rep_id": "r1", "capacity": 1}],
        },
        headers={"X-API-Key": company_api_key},
    )
    assert match_resp.status_code == 200
    assert match_resp.json()["used_default_scoring"] is False


def test_train_acepta_dataset_con_punto_y_coma(tmp_path, monkeypatch):
    """El entrenamiento lee el CSV con el mismo detector de separador que el upload."""
    import model_registry as mr
    from app.cases.lead_match import train as train_mod

    monkeypatch.setattr(train_mod.settings, "MODELS_DIR", str(tmp_path / "models"))
    import pathlib

    synthetic = pathlib.Path(__file__).resolve().parents[1] / "synthetic_data" / "lead_match.csv"
    src = synthetic.read_text(encoding="utf-8")
    path = tmp_path / "leads_excel.csv"
    path.write_text(src.replace(",", ";"), encoding="utf-8-sig")

    version_id, metrics = train_mod.train_for_company(99, str(path))
    assert metrics["n_train"] > 0
    assert mr.get_current_version_id(str(tmp_path / "models"), 99, "lead_match") == version_id
