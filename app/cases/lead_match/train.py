"""
Entrena el clasificador de conversión de LeadMatch a partir del dataset
histórico de la empresa (columnas: lead_id, origin, time_on_site_minutes,
company_size, rep_id, converted -- ver EXPECTED_COLUMNS en
app/services/dataset_ingestion.py).

Entrenar una regresión logística sobre unos pocos miles de leads es cuestión
de segundos, así que no hace falta una cola de tareas: el endpoint /train
corre sincrónico y devuelve el resultado en la misma respuesta.

Cada entrenamiento guarda una versión, pero solo pasa a ser la vigente si es útil y no
es peor que la actual (ver `_decide_promotion`). Las dos se comparan sobre las mismas
filas de prueba: las métricas de datasets distintos no son comparables entre sí.
"""

import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import model_registry
from app.services.dataset_ingestion import read_dataset_csv
from config import get_settings

settings = get_settings()
CASE = "lead_match"

FEATURE_COLUMNS = ["origin", "time_on_site_minutes", "company_size"]
NUMERIC_COLUMNS = ["time_on_site_minutes", "company_size"]
TARGET_COLUMN = "converted"

_POSITIVE_VALUES = {"1", "1.0", "true", "si", "sí", "s", "yes", "y"}
_NEGATIVE_VALUES = {"0", "0.0", "false", "no", "n"}

# Por debajo de esto, un modelo apenas distingue mejor que el azar (0.5).
MIN_PROMOTION_AUC = 0.55
# Margen para que el ruido de una muestra de prueba chica no frene un reentrenamiento.
PROMOTION_AUC_TOLERANCE = 0.01


def _build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[("origin_ohe", OneHotEncoder(handle_unknown="ignore"), ["origin"])],
        remainder="passthrough",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


class InsufficientTrainingDataError(ValueError):
    """El dataset no se puede usar para entrenar: faltan columnas o filas, no hay
    variedad en `converted`, o hay valores con un formato que no se puede interpretar."""


def _examples(values: pd.Series) -> str:
    return ", ".join(repr(v) for v in sorted(values.astype(str).unique())[:5])


def _normalize_target(series: pd.Series) -> pd.Series:
    """Convierte `converted` a 0/1, aceptando las formas habituales de un export de CRM."""
    normalized = series.astype(str).str.strip().str.lower()
    invalid = normalized[~normalized.isin(_POSITIVE_VALUES | _NEGATIVE_VALUES)]
    if not invalid.empty:
        raise InsufficientTrainingDataError(
            f"La columna '{TARGET_COLUMN}' debe indicar si el lead compró (1/0, sí/no, yes/no o "
            f"true/false). Valores no reconocidos: {_examples(invalid)}."
        )
    return normalized.isin(_POSITIVE_VALUES).astype(int)


def _to_numeric(series: pd.Series, column: str) -> pd.Series:
    """Convierte a número aceptando coma decimal, como la de un Excel en español."""
    text = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    converted = pd.to_numeric(text, errors="coerce")
    invalid = series[~np.isfinite(converted)]
    if not invalid.empty:
        raise InsufficientTrainingDataError(
            f"La columna '{column}' debe ser numérica. Valores no reconocidos: {_examples(invalid)}."
        )
    return converted


def _auc_of(model, X: pd.DataFrame, y: pd.Series) -> float:
    proba_positive = model.predict_proba(X)[:, list(model.classes_).index(1)]
    return round(float(roc_auc_score(y, proba_positive)), 4)


def _current_model_auc(company_id: int, X_test: pd.DataFrame, y_test: pd.Series) -> float | None:
    path = model_registry.get_current_artifact_path(settings.MODELS_DIR, company_id, CASE, "model.joblib")
    return None if path is None else _auc_of(joblib.load(path), X_test, y_test)


def _decide_promotion(new_auc: float | None, current_auc: float | None) -> tuple[bool, str]:
    if new_auc is None:
        return False, "No se pudo evaluar el modelo: las filas de prueba tienen un solo resultado de 'converted'."
    if new_auc < MIN_PROMOTION_AUC:
        return False, f"El modelo casi no distingue mejor que el azar (AUC {new_auc}, mínimo {MIN_PROMOTION_AUC})."
    if current_auc is None:
        return True, f"Primer modelo útil de la empresa (AUC {new_auc})."
    if new_auc < current_auc - PROMOTION_AUC_TOLERANCE:
        return False, f"Es peor que el vigente sobre las mismas filas (AUC {new_auc} contra {current_auc})."
    return True, f"No es peor que el vigente sobre las mismas filas (AUC {new_auc} contra {current_auc})."


def train_for_company(
    company_id: int, dataset_path: str, dataset_id: int | None = None
) -> tuple[str, dict, bool, str]:
    """Entrena y guarda una versión nueva. Devuelve (version_id, metrics, promoted, reason):
    `promoted` indica si quedó como vigente y `reason` explica por qué."""
    try:
        df = read_dataset_csv(dataset_path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise InsufficientTrainingDataError(f"No se pudo leer el dataset: {exc}") from exc

    missing = [c for c in [*FEATURE_COLUMNS, TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise InsufficientTrainingDataError(f"Al dataset le faltan columnas requeridas: {missing}")

    df = df.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN])
    y = _normalize_target(df[TARGET_COLUMN])
    if len(df) < 20 or y.nunique() < 2:
        raise InsufficientTrainingDataError(
            "Hacen falta al menos 20 leads con ambas clases de 'converted' representadas "
            f"para entrenar (hay {len(df)} filas, {y.nunique()} clase(s))."
        )

    X = pd.DataFrame(
        {
            "origin": df["origin"].astype(str),
            **{column: _to_numeric(df[column], column) for column in NUMERIC_COLUMNS},
        }
    )[FEATURE_COLUMNS]

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics: dict = {"n_train": len(X_train), "n_test": len(X_test), "dataset_id": dataset_id}
    new_auc = current_auc = None
    # Si y_test terminó con una sola clase tras el split, AUC no está definido --
    # se omite la métrica en vez de forzar un valor sin sentido.
    if y_test.nunique() > 1:
        new_auc = _auc_of(pipeline, X_test, y_test)
        current_auc = _current_model_auc(company_id, X_test, y_test)
        metrics["auc"] = new_auc
        if current_auc is not None:
            metrics["current_model_auc"] = current_auc

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        joblib.dump(pipeline, tmp.name)
        model_path = tmp.name

    version_id = model_registry.new_version_id("logreg")
    model_registry.save_version(
        settings.MODELS_DIR,
        company_id,
        CASE,
        version_id,
        artifacts={"model.joblib": model_path},
        metrics=metrics,
    )

    promoted, reason = _decide_promotion(new_auc, current_auc)
    if promoted:
        model_registry.promote(settings.MODELS_DIR, company_id, CASE, version_id)

    return version_id, metrics, promoted, reason
