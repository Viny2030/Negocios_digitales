"""
Registro de versiones de modelos entrenados, por empresa y por caso.

Antes de esto, cada `scripts/train_caseN.py` sobreescribía directamente el artefacto
que sirve `app/cases/caseN_*/service.py`, sin historial ni forma de bloquear un modelo
problemático. Esto tenía además un bug real: `train_case1.py`, `train_case2.py` y
`train_case4.py` guardaban con el nombre `model_{algoritmo}.zip`, mientras que los
servicios cargaban siempre `model.zip` — ningún modelo recién entrenado en esos tres
casos llegaba a servirse.

Estructura en disco (bajo settings.MODELS_DIR):
    {company_id}/{case}/versions/{version_id}/model.zip        (+ otros artefactos)
    {company_id}/{case}/versions/{version_id}/metadata.json
    {company_id}/{case}/current.json   -> puntero a la versión activa (servida)

Cada versión queda guardada siempre, incluso si no se promueve (por ejemplo, si no pasa
el filtro de equidad de case2): así queda constancia de qué se entrenó y por qué se
rechazó, y la versión anterior sigue sirviendo hasta que una nueva la reemplace.
"""
import datetime as dt
import json
import os
import shutil


def _case_dir(models_dir: str, company_id: int, case: str) -> str:
    return os.path.join(models_dir, str(company_id), case)


def _versions_dir(models_dir: str, company_id: int, case: str) -> str:
    return os.path.join(_case_dir(models_dir, company_id, case), "versions")


def _current_pointer_path(models_dir: str, company_id: int, case: str) -> str:
    return os.path.join(_case_dir(models_dir, company_id, case), "current.json")


def new_version_id(algorithm: str = "model") -> str:
    """Identificador único y ordenable cronológicamente: 20260727T143000123456Z_ppo."""
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{ts}_{algorithm.lower()}"


def save_version(
    models_dir: str, company_id: int, case: str, version_id: str,
    artifacts: dict[str, str], metrics: dict,
) -> str:
    """Crea la carpeta de la versión, mueve los artefactos entrenados (ya escritos en disco,
    p.ej. por `model.save(tmp_path)`) a su ubicación final dentro de la versión, y escribe
    metadata.json. `artifacts` mapea nombre_final -> ruta_actual_en_disco.
    Devuelve la ruta a la carpeta de la versión. No promueve la versión como "current"."""
    vdir = os.path.join(_versions_dir(models_dir, company_id, case), version_id)
    os.makedirs(vdir, exist_ok=True)

    saved_names = []
    for dest_name, src_path in artifacts.items():
        dest_path = os.path.join(vdir, dest_name)
        if os.path.abspath(src_path) != os.path.abspath(dest_path):
            shutil.move(src_path, dest_path)
        saved_names.append(dest_name)

    metadata = {
        "version_id": version_id,
        "case": case,
        "company_id": company_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "artifacts": saved_names,
        "metrics": metrics,
    }
    with open(os.path.join(vdir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return vdir


def promote(models_dir: str, company_id: int, case: str, version_id: str) -> None:
    """Marca `version_id` como la versión activa que deben servir los endpoints de inferencia."""
    vdir = os.path.join(_versions_dir(models_dir, company_id, case), version_id)
    if not os.path.isdir(vdir):
        raise ValueError(f"Versión '{version_id}' no existe para company_id={company_id}, case={case}")

    pointer = {"version_id": version_id, "promoted_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    case_dir = _case_dir(models_dir, company_id, case)
    os.makedirs(case_dir, exist_ok=True)
    with open(_current_pointer_path(models_dir, company_id, case), "w", encoding="utf-8") as f:
        json.dump(pointer, f, indent=2)


def get_current_version_id(models_dir: str, company_id: int, case: str) -> str | None:
    """Lectura barata (sin construir rutas ni chequear artefactos) usada por los servicios
    para detectar si hay una versión más nueva promovida, incluso cuando el entrenamiento
    corrió en otro proceso (p.ej. un worker de Celery, ver app/tasks.py) y por lo tanto
    `invalidate_policy_cache` nunca se ejecutó en el proceso que sirve la API."""
    pointer_path = _current_pointer_path(models_dir, company_id, case)
    if not os.path.exists(pointer_path):
        return None
    with open(pointer_path, encoding="utf-8") as f:
        return json.load(f).get("version_id")


def get_current_version_dir(models_dir: str, company_id: int, case: str) -> str | None:
    pointer_path = _current_pointer_path(models_dir, company_id, case)
    if not os.path.exists(pointer_path):
        return None
    with open(pointer_path, encoding="utf-8") as f:
        pointer = json.load(f)
    vdir = os.path.join(_versions_dir(models_dir, company_id, case), pointer["version_id"])
    return vdir if os.path.isdir(vdir) else None


def get_current_artifact_path(models_dir: str, company_id: int, case: str, filename: str) -> str | None:
    """Ruta al artefacto (p.ej. 'model.zip') de la versión actualmente promovida, o None
    si todavía no hay ninguna versión aprobada/servible (los `PolicyWrapper` de cada caso
    ya saben caer a su heurística cuando reciben model_path=None)."""
    vdir = get_current_version_dir(models_dir, company_id, case)
    if vdir is None:
        return None
    path = os.path.join(vdir, filename)
    return path if os.path.exists(path) else None


def list_versions(models_dir: str, company_id: int, case: str) -> list[dict]:
    """Historial completo de versiones (aprobadas o no), más recientes al final."""
    root = _versions_dir(models_dir, company_id, case)
    if not os.path.isdir(root):
        return []
    versions = []
    for version_id in sorted(os.listdir(root)):
        meta_path = os.path.join(root, version_id, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                versions.append(json.load(f))
    return versions
