"""
Tests de model_registry.py: versionado y promoción de modelos entrenados.

No dependen del cliente FastAPI ni de la DB -- las funciones de model_registry
trabajan directo sobre `models_dir` (un directorio), así que alcanza con
`tmp_path` de pytest para aislar cada test.
"""

import os

import pytest

import model_registry


def _make_version(models_dir, company_id, case, tag):
    """Crea un artefacto falso y lo guarda como una nueva versión (sin promoverla)."""
    src = os.path.join(models_dir, f"_src_{tag}.zip")
    with open(src, "wb") as f:
        f.write(tag.encode())
    version_id = model_registry.new_version_id(tag)
    model_registry.save_version(
        models_dir,
        company_id,
        case,
        version_id,
        artifacts={"model.zip": src},
        metrics={"tag": tag},
    )
    return version_id


def test_save_version_no_promueve_automaticamente(tmp_path):
    models_dir = str(tmp_path)
    company_id, case = 1, "case2_pricing"

    version_id = _make_version(models_dir, company_id, case, "v1")

    # El artefacto quedó guardado en su carpeta de versión...
    versions = model_registry.list_versions(models_dir, company_id, case)
    assert len(versions) == 1
    assert versions[0]["version_id"] == version_id

    # ...pero nadie lo está sirviendo todavía: get_current_* devuelve None.
    assert model_registry.get_current_version_id(models_dir, company_id, case) is None
    assert model_registry.get_current_version_dir(models_dir, company_id, case) is None
    assert model_registry.get_current_artifact_path(models_dir, company_id, case, "model.zip") is None


def test_promote_expone_la_version_como_current(tmp_path):
    models_dir = str(tmp_path)
    company_id, case = 2, "case1_messaging"

    version_id = _make_version(models_dir, company_id, case, "v1")
    model_registry.promote(models_dir, company_id, case, version_id)

    assert model_registry.get_current_version_id(models_dir, company_id, case) == version_id
    artifact_path = model_registry.get_current_artifact_path(models_dir, company_id, case, "model.zip")
    assert artifact_path is not None
    assert os.path.isfile(artifact_path)


def test_promote_version_inexistente_falla(tmp_path):
    with pytest.raises(ValueError):
        model_registry.promote(str(tmp_path), 3, "case2_pricing", "version-que-no-existe")


def test_version_no_promovida_no_reemplaza_a_la_actual(tmp_path):
    """Simula el gate de equidad de case2: una v2 que no pasa el filtro se guarda para
    historial, pero la empresa sigue sirviendo v1 hasta que alguien promueva otra cosa."""
    models_dir = str(tmp_path)
    company_id, case = 4, "case2_pricing"

    v1 = _make_version(models_dir, company_id, case, "v1_aprobada")
    model_registry.promote(models_dir, company_id, case, v1)

    v2 = _make_version(models_dir, company_id, case, "v2_rechazada_por_fairness")
    # v2 nunca se promueve (equivalente a approved=False en scripts/train_case2.py)

    assert model_registry.get_current_version_id(models_dir, company_id, case) == v1

    versions = model_registry.list_versions(models_dir, company_id, case)
    assert {v["version_id"] for v in versions} == {v1, v2}


def test_rollback_manual_a_una_version_anterior(tmp_path):
    """Si una versión promovida resulta problemática, promover una versión anterior
    (rollback manual) debe volver a dejarla como la servida."""
    models_dir = str(tmp_path)
    company_id, case = 5, "case3_anomaly"

    v1 = _make_version(models_dir, company_id, case, "v1")
    model_registry.promote(models_dir, company_id, case, v1)

    v2 = _make_version(models_dir, company_id, case, "v2_problematica")
    model_registry.promote(models_dir, company_id, case, v2)
    assert model_registry.get_current_version_id(models_dir, company_id, case) == v2

    # Rollback: volver a promover v1
    model_registry.promote(models_dir, company_id, case, v1)
    assert model_registry.get_current_version_id(models_dir, company_id, case) == v1


def test_companies_y_casos_distintos_estan_aislados(tmp_path):
    """Una versión de la empresa 1 no debe filtrarse a la empresa 2, ni un caso a otro."""
    models_dir = str(tmp_path)

    v_company1 = _make_version(models_dir, 1, "case1_messaging", "empresa1")
    model_registry.promote(models_dir, 1, "case1_messaging", v_company1)

    assert model_registry.get_current_version_id(models_dir, 2, "case1_messaging") is None
    assert model_registry.get_current_version_id(models_dir, 1, "case2_pricing") is None
    assert model_registry.get_current_version_id(models_dir, 1, "case1_messaging") == v_company1


def test_get_current_artifact_path_filename_inexistente_devuelve_none(tmp_path):
    """Si la versión promovida no incluye el artefacto pedido (ej. fp_classifier.joblib
    en un caso que no lo generó), no debe explotar -- solo devolver None."""
    models_dir = str(tmp_path)
    company_id, case = 6, "case3_anomaly"

    v1 = _make_version(models_dir, company_id, case, "sin_clasificador")
    model_registry.promote(models_dir, company_id, case, v1)

    assert model_registry.get_current_artifact_path(models_dir, company_id, case, "model.zip") is not None
    assert model_registry.get_current_artifact_path(models_dir, company_id, case, "fp_classifier.joblib") is None
