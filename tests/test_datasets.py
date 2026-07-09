import io


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

    resp_list = client.get("/api/v1/datasets/case1_messaging", headers={"X-API-Key": company_api_key})
    assert resp_list.status_code == 200
    assert len(resp_list.json()) == 1
