"""
Tests de app/api/v1/endpoints/datasets.py + app/services/dataset_ingestion.py:
upload, listado, reemplazo in-place, activación, borrado y las dos reglas nuevas
de diseño (dedupe por hash de contenido y "dataset activo" por empresa+caso).
"""

import io


def _csv_bytes(rows: int, base_value: str = "email_a") -> bytes:
    header = b"prospect_id,channels_contacted,converted\n"
    body = b"".join(f"p-{i},{base_value},{i % 2}\n".encode() for i in range(rows))
    return header + body


def test_upload_and_list_dataset(client, company_api_key):
    csv_content = b"prospect_id,channels_contacted,converted\np-1,email_a,1\np-2,sms,0\n"
    files = {"file": ("campanas.csv", io.BytesIO(csv_content), "text/csv")}

    resp = client.post(
        "/api/v1/datasets/case1_messaging",
        files=files,
        headers={"X-API-Key": company_api_key},
    )
    assert resp.status_code == 201
    dataset = resp.json()
    assert dataset["row_count"] == 2
    assert dataset["is_active"] is True

    resp_list = client.get("/api/v1/datasets/case1_messaging", headers={"X-API-Key": company_api_key})
    assert resp_list.status_code == 200
    assert len(resp_list.json()) == 1


def test_row_count_no_se_trunca_a_5000(client, company_api_key):
    """Antes `row_count` salía de `len(pd.read_csv(path, nrows=5000))`, así que un
    dataset con más de 5000 filas reales quedaba reportado como 5000."""
    files = {"file": ("grande.csv", io.BytesIO(_csv_bytes(6000)), "text/csv")}
    resp = client.post(
        "/api/v1/datasets/case1_messaging",
        files=files,
        headers={"X-API-Key": company_api_key},
    )
    assert resp.status_code == 201
    assert resp.json()["row_count"] == 6000


def test_upload_mismo_contenido_no_duplica(client, company_api_key):
    content = _csv_bytes(10)
    headers = {"X-API-Key": company_api_key}

    first = client.post(
        "/api/v1/datasets/case1_messaging",
        files={"file": ("a.csv", io.BytesIO(content), "text/csv")},
        headers=headers,
    )
    second = client.post(
        "/api/v1/datasets/case1_messaging",
        files={"file": ("a_copia.csv", io.BytesIO(content), "text/csv")},
        headers=headers,
    )

    assert first.status_code == 201 and second.status_code == 201
    assert first.json()["id"] == second.json()["id"]

    listed = client.get("/api/v1/datasets/case1_messaging", headers=headers)
    assert len(listed.json()) == 1


def test_upload_contenido_distinto_desactiva_el_anterior(client, company_api_key):
    headers = {"X-API-Key": company_api_key}

    first = client.post(
        "/api/v1/datasets/case1_messaging",
        files={"file": ("v1.csv", io.BytesIO(_csv_bytes(5, "email_a")), "text/csv")},
        headers=headers,
    )
    second = client.post(
        "/api/v1/datasets/case1_messaging",
        files={"file": ("v2.csv", io.BytesIO(_csv_bytes(5, "sms")), "text/csv")},
        headers=headers,
    )
    assert first.json()["id"] != second.json()["id"]
    assert second.json()["is_active"] is True

    listed = {d["id"]: d for d in client.get("/api/v1/datasets/case1_messaging", headers=headers).json()}
    assert listed[first.json()["id"]]["is_active"] is False
    assert listed[second.json()["id"]]["is_active"] is True


def test_activate_dataset_vuelve_a_marcar_uno_anterior_como_vigente(client, company_api_key):
    headers = {"X-API-Key": company_api_key}

    first = client.post(
        "/api/v1/datasets/case1_messaging",
        files={"file": ("v1.csv", io.BytesIO(_csv_bytes(5, "email_a")), "text/csv")},
        headers=headers,
    ).json()
    second = client.post(
        "/api/v1/datasets/case1_messaging",
        files={"file": ("v2.csv", io.BytesIO(_csv_bytes(5, "sms")), "text/csv")},
        headers=headers,
    ).json()
    assert second["is_active"] is True

    resp = client.post(
        f"/api/v1/datasets/case1_messaging/{first['id']}/activate",
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["is_active"] is True

    listed = {d["id"]: d for d in client.get("/api/v1/datasets/case1_messaging", headers=headers).json()}
    assert listed[first["id"]]["is_active"] is True
    assert listed[second["id"]]["is_active"] is False


def test_replace_dataset_mantiene_el_mismo_id(client, company_api_key):
    headers = {"X-API-Key": company_api_key}

    original = client.post(
        "/api/v1/datasets/case1_messaging",
        files={"file": ("v1.csv", io.BytesIO(_csv_bytes(3)), "text/csv")},
        headers=headers,
    ).json()

    replaced = client.put(
        f"/api/v1/datasets/case1_messaging/{original['id']}",
        files={"file": ("corregido.csv", io.BytesIO(_csv_bytes(9)), "text/csv")},
        headers=headers,
    )
    assert replaced.status_code == 200
    body = replaced.json()
    assert body["id"] == original["id"]
    assert body["row_count"] == 9
    assert body["name"] == "corregido.csv"
    assert body["updated_at"] is not None
    assert body["is_active"] is True

    listed = client.get("/api/v1/datasets/case1_messaging", headers=headers).json()
    assert len(listed) == 1  # sigue siendo un solo dataset, no se creó uno nuevo


def test_replace_dataset_de_otra_empresa_da_404(client, company_api_key):
    headers = {"X-API-Key": company_api_key}
    resp = client.put(
        "/api/v1/datasets/case1_messaging/999999",
        files={"file": ("x.csv", io.BytesIO(_csv_bytes(1)), "text/csv")},
        headers=headers,
    )
    assert resp.status_code == 404


def test_upload_rechaza_csv_de_otro_caso(client, company_api_key):
    """El CSV de case2_pricing (product_id, base_cost, ...) no trae ninguna columna
    esperada para case1_messaging -- antes se aceptaba igual como 'validated'."""
    case2_csv = (
        b"product_id,base_cost,user_segment,inventory_level_pct,competitor_avg_price,demand_index\n"
        b"sku-1,100.0,premium,50.0,150.0,0.5\n"
    )
    resp = client.post(
        "/api/v1/datasets/case1_messaging",
        files={"file": ("por_error.csv", io.BytesIO(case2_csv), "text/csv")},
        headers={"X-API-Key": company_api_key},
    )
    assert resp.status_code == 400
    assert "case1_messaging" in resp.json()["detail"]

    listed = client.get("/api/v1/datasets/case1_messaging", headers={"X-API-Key": company_api_key}).json()
    assert listed == []  # no queda ni el registro ni el archivo huérfano


def test_replace_dataset_rechaza_csv_de_otro_caso(client, company_api_key):
    headers = {"X-API-Key": company_api_key}
    original = client.post(
        "/api/v1/datasets/case1_messaging",
        files={"file": ("v1.csv", io.BytesIO(_csv_bytes(3)), "text/csv")},
        headers=headers,
    ).json()

    case2_csv = b"product_id,base_cost,user_segment\nsku-1,100.0,premium\n"
    resp = client.put(
        f"/api/v1/datasets/case1_messaging/{original['id']}",
        files={"file": ("por_error.csv", io.BytesIO(case2_csv), "text/csv")},
        headers=headers,
    )
    assert resp.status_code == 400

    # el dataset original no se tocó
    listed = client.get("/api/v1/datasets/case1_messaging", headers=headers).json()
    assert listed[0]["row_count"] == 3


def test_delete_dataset_lo_saca_de_la_lista(client, company_api_key):
    headers = {"X-API-Key": company_api_key}
    dataset = client.post(
        "/api/v1/datasets/case1_messaging",
        files={"file": ("a.csv", io.BytesIO(_csv_bytes(2)), "text/csv")},
        headers=headers,
    ).json()

    resp = client.delete(f"/api/v1/datasets/case1_messaging/{dataset['id']}", headers=headers)
    assert resp.status_code == 204

    listed = client.get("/api/v1/datasets/case1_messaging", headers=headers).json()
    assert listed == []
