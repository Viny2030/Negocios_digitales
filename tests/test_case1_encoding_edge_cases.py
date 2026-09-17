"""
Casos límite del Caso 1.1 que antes rompían en producción, cubiertos por separado de
test_encoding.py (que otro agente puede estar tocando en paralelo para case2):

1. `record_to_profile`/`row_to_vector` con NaN en columnas numéricas/booleanas no debe
   crashear (antes `record.get(key, default) or default` dejaba pasar NaN tal cual --
   es "truthy" en Python -- y `int(nan)` levanta ValueError; `bool(nan)` da True,
   marcando por error un prospecto como unsubscribed).
2. Un `0` real (no NaN, no ausente) en esas mismas columnas debe conservarse, no
   reemplazarse silenciosamente por el default.
3. `MessagingPolicy` debe elegir PPO/DQN según el algoritmo recibido explícitamente
   (leído de los metadatos de la versión promovida por model_registry), no adivinando
   por el nombre del artefacto -- que siempre es "model.zip".
4. `POST /case1/train` debe devolver 404 si el dataset no existe o no pertenece a la
   empresa autenticada, en vez de encolar igual y fallar recién dentro del worker.
"""

import numpy as np
import pandas as pd

import app.cases.case1_messaging.policy as policy_module
import model_registry
from app.cases.case1_messaging.encoding import _bool, _num, record_to_profile, row_to_vector

# --- 0) Los helpers NaN-safe en sí mismos: un 0/False real no es un NaN/ausente ------


def test_num_helper_preserves_real_zero_over_nonzero_default():
    """Con la lógica vieja (`row.get(key, default) or default`), un 0 real hubiera
    caído al default porque 0 es 'falsy' en Python -- acá el default es distinto de 0
    justamente para que la diferencia sea observable."""
    row = {"days_since_last_contact": 0}
    assert _num(row, "days_since_last_contact", 7.0) == 0.0


def test_num_helper_falls_back_to_default_on_nan():
    row = {"days_since_last_contact": float("nan")}
    assert _num(row, "days_since_last_contact", 7.0) == 7.0


def test_num_helper_falls_back_to_default_when_missing():
    assert _num({}, "days_since_last_contact", 7.0) == 7.0


def test_bool_helper_preserves_real_false_over_true_default():
    row = {"unsubscribed": False}
    assert _bool(row, "unsubscribed", True) is False


def test_bool_helper_falls_back_to_default_on_nan():
    """Antes, `bool(float('nan'))` da True: un NaN se leía como 'sí, unsubscribed'."""
    row = {"unsubscribed": float("nan")}
    assert _bool(row, "unsubscribed", True) is True


# --- 1) NaN en columnas numéricas/booleanas no crashea, cae al default ---------------


def test_record_to_profile_nan_numeric_fields_falls_back_to_defaults():
    record = {
        "prospect_id": "p-nan",
        "channels_contacted": "email_a",
        "messages_received": float("nan"),
        "responses_count": float("nan"),
        "days_since_last_contact": float("nan"),
        "purchase_stage": "consideration",
        "unsubscribed": float("nan"),
    }

    profile = record_to_profile(record)  # no debe levantar ValueError

    assert profile.messages_received == 0
    assert profile.responses_count == 0
    assert profile.days_since_last_contact == 0.0
    assert profile.unsubscribed is False  # NaN no debe leerse como "truthy" -> unsubscribed


def test_row_to_vector_nan_numeric_fields_does_not_crash():
    row = pd.Series(
        {
            "prospect_id": "p-nan-row",
            "channels_contacted": "sms",
            "messages_received": np.nan,
            "responses_count": np.nan,
            "days_since_last_contact": np.nan,
            "purchase_stage": np.nan,  # también puede venir NaN, no solo string vacío
            "unsubscribed": np.nan,
        }
    )

    vector = row_to_vector(row)

    assert vector.shape == (14,)
    assert np.isfinite(vector).all()
    # purchase_stage NaN -> cae a "awareness" (offset 10, primer elemento del one-hot de etapa)
    assert vector[10] == 1.0
    assert vector[-1] == 0.0  # unsubscribed NaN -> False, no True


# --- 2) Un 0 real se conserva, no se pisa con el default -----------------------------


def test_record_to_profile_real_zero_is_preserved_not_replaced_by_default():
    record = {
        "prospect_id": "p-zero",
        "channels_contacted": "",
        "messages_received": 0,
        "responses_count": 0,
        "days_since_last_contact": 0,
        "purchase_stage": "awareness",
        "unsubscribed": False,
    }

    profile = record_to_profile(record)

    assert profile.messages_received == 0
    assert profile.responses_count == 0
    assert profile.days_since_last_contact == 0.0
    assert profile.unsubscribed is False


def test_row_to_vector_real_zero_days_since_last_contact_is_preserved():
    """`days_since_last_contact=0` (contacto justo hoy) es un valor real y distinto de
    'sin dato' -- antes del fix, si algún día se hubiera usado `or default` acá, un 0
    real se hubiera confundido con ausencia de dato."""
    row = pd.Series(
        {
            "prospect_id": "p-today",
            "channels_contacted": "email_a",
            "messages_received": 0,
            "responses_count": 0,
            "days_since_last_contact": 0,
            "purchase_stage": "awareness",
            "unsubscribed": False,
        }
    )
    vector = row_to_vector(row)
    assert vector[9] == 0.0  # days_since_last_contact / 30 == 0.0, no el default distinto


# --- 3) model_registry.get_current_metrics + selección de algoritmo en MessagingPolicy


def test_get_current_metrics_reads_algorithm_from_promoted_version(tmp_path):
    models_dir = str(tmp_path)
    company_id = 999
    case = "case1_messaging"
    version_id = model_registry.new_version_id("ppo")

    artifact_path = tmp_path / "_tmp_model.zip"
    artifact_path.write_text("fake-model-bytes")

    model_registry.save_version(
        models_dir,
        company_id,
        case,
        version_id,
        artifacts={"model.zip": str(artifact_path)},
        metrics={"algorithm": "ppo", "timesteps": 1000, "approved": True},
    )
    model_registry.promote(models_dir, company_id, case, version_id)

    metrics = model_registry.get_current_metrics(models_dir, company_id, case)
    assert metrics is not None
    assert metrics["algorithm"] == "ppo"


def test_get_current_metrics_returns_none_without_promoted_version(tmp_path):
    assert model_registry.get_current_metrics(str(tmp_path), 12345, "case1_messaging") is None


class _FakeAlgo:
    """Doble de DQN/PPO: no depende de stable-baselines3 real ni de un modelo entrenado
    de verdad, solo de que `MessagingPolicy` elija la CLASE correcta según `algorithm`."""

    def __init__(self, name: str):
        self.name = name

    @classmethod
    def load(cls, path):
        return cls(name=cls._label)


def _make_fake_algo_class(label: str):
    return type(f"Fake{label}", (_FakeAlgo,), {"_label": label})


def test_messaging_policy_uses_ppo_when_metadata_says_ppo(tmp_path, monkeypatch):
    fake_dqn = _make_fake_algo_class("DQN")
    fake_ppo = _make_fake_algo_class("PPO")
    monkeypatch.setattr(policy_module, "DQN", fake_dqn)
    monkeypatch.setattr(policy_module, "PPO", fake_ppo)

    model_path = tmp_path / "model.zip"  # nombre siempre genérico, sin "ppo"/"dqn"
    model_path.write_text("fake")

    policy = policy_module.MessagingPolicy(model_path=str(model_path), algorithm="ppo")

    assert policy.model.name == "PPO"


def test_messaging_policy_defaults_to_dqn_when_no_algorithm_given(tmp_path, monkeypatch):
    """Sin metadata (algorithm=None) el comportamiento histórico (DQN) se conserva."""
    fake_dqn = _make_fake_algo_class("DQN")
    fake_ppo = _make_fake_algo_class("PPO")
    monkeypatch.setattr(policy_module, "DQN", fake_dqn)
    monkeypatch.setattr(policy_module, "PPO", fake_ppo)

    model_path = tmp_path / "model.zip"
    model_path.write_text("fake")

    policy = policy_module.MessagingPolicy(model_path=str(model_path), algorithm=None)

    assert policy.model.name == "DQN"


# --- 4) Ownership check antes de encolar el entrenamiento -----------------------------


def test_train_endpoint_returns_404_for_nonexistent_dataset(client, company_api_key):
    resp = client.post(
        "/api/v1/case1/train",
        json={"dataset_id": 999999, "algorithm": "dqn", "timesteps": 1000},
        headers={"X-API-Key": company_api_key},
    )
    assert resp.status_code == 404


def test_train_endpoint_returns_404_for_dataset_owned_by_another_company(client, company_api_key):
    import uuid

    other_email = f"otra-{uuid.uuid4().hex[:8]}@empresa.com"
    other_resp = client.post("/api/v1/companies", json={"name": "Otra Empresa", "contact_email": other_email})
    assert other_resp.status_code == 201
    other_api_key = other_resp.json()["api_key"]

    csv_content = (
        "prospect_id,channels_contacted,messages_received,responses_count,"
        "days_since_last_contact,purchase_stage,unsubscribed\n"
        "p-1,email_a,1,0,5,awareness,False\n"
    )
    upload_resp = client.post(
        "/api/v1/datasets/case1_messaging",
        files={"file": ("prospects.csv", csv_content, "text/csv")},
        headers={"X-API-Key": other_api_key},
    )
    assert upload_resp.status_code == 201
    other_dataset_id = upload_resp.json()["id"]

    resp = client.post(
        "/api/v1/case1/train",
        json={"dataset_id": other_dataset_id, "algorithm": "dqn", "timesteps": 1000},
        headers={"X-API-Key": company_api_key},
    )
    assert resp.status_code == 404
