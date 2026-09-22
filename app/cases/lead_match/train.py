"""
Entrena el clasificador de conversión de LeadMatch a partir del dataset
histórico de la empresa (columnas: lead_id, origin, time_on_site_minutes,
company_size, rep_id, converted -- ver EXPECTED_COLUMNS en
app/services/dataset_ingestion.py).

A diferencia de los casos RL de la plataforma anterior, entrenar una regresión
logística sobre unos pocos miles de leads es cuestión de segundos -- no hace
falta encolarlo en Celery como los entrenamientos viejos (ver app/tasks.py):
el endpoint /train de este caso corre sincrónico y devuelve el resultado en
la misma respuesta.
"""

import tempfile

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import model_registry
from config import get_settings

settings = get_settings()
CASE = "lead_match"

FEATURE_COLUMNS = ["origin", "time_on_site_minutes", "company_size"]
TARGET_COLUMN = "converted"


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
    """El dataset no trae suficientes filas o suficiente variedad de la clase
    `converted` (ej. todos los leads convirtieron, o ninguno) para entrenar un
    clasificador con algo de validez estadística."""


def train_for_company(company_id: int, dataset_path: str) -> tuple[str, dict]:
    df = pd.read_csv(dataset_path)
    missing = [c for c in [*FEATURE_COLUMNS, TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise InsufficientTrainingDataError(f"Al dataset le faltan columnas requeridas: {missing}")

    df = df.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN])
    if len(df) < 20 or df[TARGET_COLUMN].nunique() < 2:
        raise InsufficientTrainingDataError(
            "Hacen falta al menos 20 leads con ambas clases de 'converted' representadas "
            f"para entrenar (hay {len(df)} filas, {df[TARGET_COLUMN].nunique()} clase(s))."
        )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    metrics: dict = {"n_train": len(X_train), "n_test": len(X_test)}
    if y_test.nunique() > 1:
        classes = list(pipeline.classes_)
        proba_positive = pipeline.predict_proba(X_test)[:, classes.index(1)]
        metrics["auc"] = round(float(roc_auc_score(y_test, proba_positive)), 4)
    # Si y_test terminó con una sola clase tras el split, AUC no está definido --
    # se omite la métrica en vez de forzar un valor sin sentido.

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
    model_registry.promote(settings.MODELS_DIR, company_id, CASE, version_id)

    return version_id, metrics
