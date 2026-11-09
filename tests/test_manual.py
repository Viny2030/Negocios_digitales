"""Los manuales se sirven, coinciden con su generador y sus grafos están completos."""

import re

import pytest

from scripts import generar_manual as gm


@pytest.mark.parametrize(("path", "lang"), [("/manual", "es"), ("/manual/en", "en")])
def test_el_manual_se_sirve_sin_api_key(client, path, lang):
    resp = client.get(path)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert f'<html lang="{lang}">' in resp.text


def test_la_raiz_lleva_al_manual(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/manual"


def test_el_favicon_no_da_404(client):
    assert client.get("/favicon.ico").status_code == 204


@pytest.mark.parametrize("lang", ["es", "en"])
def test_el_html_coincide_con_el_generador(lang):
    saved = (gm.STATIC_DIR / gm.PAGES[lang]["file"]).read_text(encoding="utf-8")
    assert saved == gm.render(lang), "Falta correr: python -m scripts.generar_manual"


@pytest.mark.parametrize("lang", ["es", "en"])
@pytest.mark.parametrize(("nodes", "key"), [(gm.FLOW_NODES, "flow"), (gm.INFO_NODES, "info")])
def test_todos_los_nodos_tienen_nombre_en_cada_idioma(lang, nodes, key):
    texts = gm.GRAPH_TEXT[lang][key]
    assert {node[0] for node in nodes} == set(texts)
    assert all(title.strip() for title, _lines in texts.values())


@pytest.mark.parametrize("lang", ["es", "en"])
def test_los_enlaces_internos_llevan_a_una_seccion(lang):
    html = gm.render(lang)
    targets = set(re.findall(r'href="#([\w-]+)"', html))
    ids = set(re.findall(r'id="([\w-]+)"', html))
    assert targets and targets <= ids


@pytest.mark.parametrize("lang", ["es", "en"])
def test_el_manual_documenta_todas_las_rutas_de_negocio(client, lang):
    html = gm.render(lang)
    paths = client.get("/openapi.json").json()["paths"]
    for path in paths:
        if path == "/health":
            continue
        documented = path.replace("{case}", "lead_match").replace("{dataset_id}", "{id}")
        if documented.startswith("/api/v1/companies/me") and documented.endswith("/me"):
            documented = "/api/v1/companies/me"
        assert documented in html, f"{path} no aparece en el manual ({lang})"
