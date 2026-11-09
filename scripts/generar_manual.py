"""
Genera los manuales de la plataforma en español e inglés, con sus dos grafos:

    app/static/manual.html      -> /manual
    app/static/manual_en.html   -> /manual/en

Los dos idiomas comparten la posición de los nodos y las conexiones, así un grafo no
puede cambiar en un idioma y quedar viejo en el otro. Después de editar este archivo:

    python -m scripts.generar_manual

tests/test_manual.py falla si los HTML no coinciden con lo que genera este script.
"""

from html import escape
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parents[1] / "app" / "static"

# ---------------------------------------------------------------------------------------
# Grafos: posición y tipo de cada nodo (id, x, y, ancho, alto, tipo) y conexiones.
# ---------------------------------------------------------------------------------------
_COL = (50, 290, 530, 770)
_ROW = {"pricing": 140, "route": 240, "upload": 340, "train": 440, "match": 540}
_W, _H = 200, 58

FLOW_NODES = [
    ("client", 30, 30, 170, _H, "input"),
    ("ip", 230, 30, 170, _H, "control"),
    ("size", 430, 30, 170, _H, "control"),
    ("company_limit", 630, 30, 170, _H, "control"),
    ("auth", 830, 30, 170, _H, "control"),
    ("pricing_ep", _COL[0], _ROW["pricing"], _W, _H, "endpoint"),
    ("elasticity", _COL[1], _ROW["pricing"], _W, _H, "process"),
    ("optimizer", _COL[2], _ROW["pricing"], _W, _H, "process"),
    ("price", _COL[3], _ROW["pricing"], _W, _H, "output"),
    ("route_ep", _COL[0], _ROW["route"], _W, _H, "endpoint"),
    ("savings", _COL[1], _ROW["route"], _W, _H, "process"),
    ("feasible", _COL[2], _ROW["route"], _W, _H, "process"),
    ("routes", _COL[3], _ROW["route"], _W, _H, "output"),
    ("upload_ep", _COL[0], _ROW["upload"], _W, _H, "endpoint"),
    ("ingest", _COL[1], _ROW["upload"], _W, _H, "process"),
    ("csv", _COL[2], _ROW["upload"], _W, _H, "store"),
    ("train_ep", _COL[0], _ROW["train"], _W, _H, "endpoint"),
    ("training", _COL[1], _ROW["train"], _W, _H, "process"),
    ("registry", _COL[2], _ROW["train"], _W, _H, "store"),
    ("match_ep", _COL[0], _ROW["match"], _W, _H, "endpoint"),
    ("scoring", _COL[1], _ROW["match"], _W, _H, "process"),
    ("hungarian", _COL[2], _ROW["match"], _W, _H, "process"),
    ("assignments", _COL[3], _ROW["match"], _W, _H, "output"),
]


def _lane_edges(row: int, count: int) -> list[str]:
    y = row + _H // 2
    return [f"M{_COL[i] + _W} {y} L{_COL[i + 1]} {y}" for i in range(count - 1)]


FLOW_EDGES = [
    f"M200 {30 + _H // 2} L230 {30 + _H // 2}",
    f"M400 {30 + _H // 2} L430 {30 + _H // 2}",
    f"M600 {30 + _H // 2} L630 {30 + _H // 2}",
    f"M800 {30 + _H // 2} L830 {30 + _H // 2}",
    # De la autenticación a cada producto, por una línea común a la izquierda.
    *[f"M915 88 L915 114 L22 114 L22 {y + _H // 2} L{_COL[0]} {y + _H // 2}" for y in _ROW.values()],
    *_lane_edges(_ROW["pricing"], 4),
    *_lane_edges(_ROW["route"], 4),
    *_lane_edges(_ROW["upload"], 3),
    *_lane_edges(_ROW["train"], 3),
    *_lane_edges(_ROW["match"], 4),
    # El entrenamiento lee el CSV y el scoring usa el modelo vigente.
    f"M{_COL[2] + 40} {_ROW['upload'] + _H} L{_COL[1] + _W - 40} {_ROW['train']}",
    f"M{_COL[2] + 40} {_ROW['train'] + _H} L{_COL[1] + _W - 40} {_ROW['match']}",
]
FLOW_LABELS = [("reads", 575, 428), ("current", 615, 528)]
FLOW_VIEWBOX = (1020, 620)

INFO_NODES = [
    ("company", 30, 30, 270, 104, "store"),
    ("dataset", 385, 30, 270, 104, "store"),
    ("csv", 740, 30, 270, 104, "store"),
    ("current", 30, 205, 270, 72, "store"),
    ("model", 385, 190, 270, 104, "store"),
    ("train_cols", 740, 190, 270, 72, "process"),
    ("unused_cols", 740, 290, 270, 72, "muted"),
    ("pricing_in", 30, 430, 300, 72, "input"),
    ("route_in", 375, 430, 300, 72, "input"),
    ("match_in", 720, 430, 300, 72, "input"),
]
INFO_EDGES = [
    "M300 82 L385 82",
    "M655 82 L740 82",
    "M875 134 L875 190",
    "M1010 82 L1032 82 L1032 326 L1010 326",
    "M740 226 L655 226",
    "M260 134 L260 170 L342 170 L342 215 L385 215",
    "M300 250 L385 250",
]
INFO_LABELS = [
    ("one_to_many", 342, 72),
    ("file", 697, 72),
    ("contains", 915, 166),
    ("trains", 697, 216),
    ("versions", 300, 160),
    ("points_to", 342, 270),
]
INFO_TITLE_POS = (30, 415)
INFO_VIEWBOX = (1050, 515)

GRAPH_TEXT = {
    "es": {
        "flow": {
            "client": ("Sistema del cliente", ["envía el pedido"]),
            "ip": ("Techo por IP", ["600 por minuto"]),
            "size": ("Tope de tamaño", ["25 MB archivo, 10 MB JSON"]),
            "company_limit": ("Límite por empresa", ["120 por minuto"]),
            "auth": ("Autenticación", ["header X-API-Key"]),
            "pricing_ep": ("DynamicPricing", ["/dynamic-pricing/price"]),
            "elasticity": ("Elasticidad", ["regresión log-log"]),
            "optimizer": ("Optimizador de precio", ["margen y variación diaria"]),
            "price": ("Precio recomendado", ["por SKU"]),
            "route_ep": ("SmartRoute", ["/smart-route/plan-routes"]),
            "savings": ("Armado de rutas", ["ahorros de Clarke-Wright"]),
            "feasible": ("Vehículos y horarios", ["2-opt y ventanas horarias"]),
            "routes": ("Rutas", ["y paradas sin asignar"]),
            "upload_ep": ("Carga de datos", ["/datasets/lead_match"]),
            "ingest": ("Ingesta", ["valida y deduplica"]),
            "csv": ("Archivo CSV", ["y registro del dataset"]),
            "train_ep": ("Entrenamiento", ["/lead-match/train"]),
            "training": ("Regresión logística", ["una por empresa"]),
            "registry": ("Registro de modelos", ["versiones y vigente"]),
            "match_ep": ("LeadMatch", ["/lead-match/match"]),
            "scoring": ("Scoring", ["modelo o heurística"]),
            "hungarian": ("Asignación", ["algoritmo húngaro"]),
            "assignments": ("Asignaciones", ["lead a vendedor"]),
        },
        "flow_labels": {"reads": "lee", "current": "modelo vigente"},
        "info": {
            "company": (
                "Empresa",
                [
                    "tabla companies",
                    "id, name, contact_email",
                    "api_key_hash, api_key_prefix",
                    "is_active, created_at",
                ],
            ),
            "dataset": (
                "Dataset",
                ["tabla datasets", "id, name, row_count, status", "content_hash, is_active", "file_path"],
            ),
            "csv": ("Archivo CSV", ["storage/datasets/<empresa>/", "lead_match/", "lo sube la empresa"]),
            "current": ("Versión vigente", ["current.json", "la que usa /lead-match/match"]),
            "model": (
                "Versión de modelo",
                ["storage/models/<empresa>/lead_match/", "versions/<versión>/", "model.joblib, metadata.json"],
            ),
            "train_cols": ("Columnas que entrenan", ["origin, time_on_site_minutes,", "company_size, converted"]),
            "unused_cols": ("Columnas sin uso al entrenar", ["lead_id, rep_id"]),
            "pricing_in": ("Datos de DynamicPricing", ["skus, purchase_events"]),
            "route_in": ("Datos de SmartRoute", ["depot, stops, vehicles"]),
            "match_in": ("Datos de LeadMatch", ["leads, reps"]),
        },
        "info_labels": {
            "one_to_many": "1 a N",
            "file": "file_path",
            "contains": "contiene",
            "trains": "entrena",
            "versions": "1 a N",
            "points_to": "apunta a",
        },
        "info_title": "Llegan en cada pedido y no se guardan",
        "legend": {
            "input": "Entrada",
            "control": "Control",
            "endpoint": "Ruta de la API",
            "process": "Proceso",
            "store": "Dato guardado",
            "output": "Respuesta",
        },
        "aria_flow": "Grafo de funcionamiento: recorrido de un pedido por los controles y los tres productos",
        "aria_info": "Grafo de información: qué datos guarda la plataforma y cuáles no",
    },
    "en": {
        "flow": {
            "client": ("Client system", ["sends the request"]),
            "ip": ("Per-IP ceiling", ["600 per minute"]),
            "size": ("Size cap", ["25 MB file, 10 MB JSON"]),
            "company_limit": ("Per-company limit", ["120 per minute"]),
            "auth": ("Authentication", ["X-API-Key header"]),
            "pricing_ep": ("DynamicPricing", ["/dynamic-pricing/price"]),
            "elasticity": ("Elasticity", ["log-log regression"]),
            "optimizer": ("Price optimizer", ["margin and daily variation"]),
            "price": ("Recommended price", ["per SKU"]),
            "route_ep": ("SmartRoute", ["/smart-route/plan-routes"]),
            "savings": ("Route building", ["Clarke-Wright savings"]),
            "feasible": ("Vehicles and schedule", ["2-opt and time windows"]),
            "routes": ("Routes", ["and unassigned stops"]),
            "upload_ep": ("Data upload", ["/datasets/lead_match"]),
            "ingest": ("Ingestion", ["validates and deduplicates"]),
            "csv": ("CSV file", ["and dataset record"]),
            "train_ep": ("Training", ["/lead-match/train"]),
            "training": ("Logistic regression", ["one per company"]),
            "registry": ("Model registry", ["versions and current"]),
            "match_ep": ("LeadMatch", ["/lead-match/match"]),
            "scoring": ("Scoring", ["model or heuristic"]),
            "hungarian": ("Assignment", ["Hungarian algorithm"]),
            "assignments": ("Assignments", ["lead to sales rep"]),
        },
        "flow_labels": {"reads": "reads", "current": "current model"},
        "info": {
            "company": (
                "Company",
                [
                    "companies table",
                    "id, name, contact_email",
                    "api_key_hash, api_key_prefix",
                    "is_active, created_at",
                ],
            ),
            "dataset": (
                "Dataset",
                ["datasets table", "id, name, row_count, status", "content_hash, is_active", "file_path"],
            ),
            "csv": ("CSV file", ["storage/datasets/<company>/", "lead_match/", "uploaded by the company"]),
            "current": ("Current version", ["current.json", "the one /lead-match/match uses"]),
            "model": (
                "Model version",
                ["storage/models/<company>/lead_match/", "versions/<version>/", "model.joblib, metadata.json"],
            ),
            "train_cols": ("Training columns", ["origin, time_on_site_minutes,", "company_size, converted"]),
            "unused_cols": ("Columns unused in training", ["lead_id, rep_id"]),
            "pricing_in": ("DynamicPricing data", ["skus, purchase_events"]),
            "route_in": ("SmartRoute data", ["depot, stops, vehicles"]),
            "match_in": ("LeadMatch data", ["leads, reps"]),
        },
        "info_labels": {
            "one_to_many": "1 to N",
            "file": "file_path",
            "contains": "contains",
            "trains": "trains",
            "versions": "1 to N",
            "points_to": "points to",
        },
        "info_title": "Sent with each request and not stored",
        "legend": {
            "input": "Input",
            "control": "Control",
            "endpoint": "API route",
            "process": "Process",
            "store": "Stored data",
            "output": "Response",
        },
        "aria_flow": "Operation graph: how a request moves through the controls and the three products",
        "aria_info": "Information graph: which data the platform stores and which it does not",
    },
}


def _node_svg(node: tuple, text: tuple[str, list[str]]) -> str:
    node_id, x, y, width, height, kind = node
    title, lines = text
    center = x + width / 2
    total = 18 + 15 * len(lines)
    first_baseline = y + (height - total) / 2 + 14
    parts = [
        f'<g class="node {kind}" data-node="{node_id}">',
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8"/>',
        f'<text class="title" x="{center:g}" y="{first_baseline:g}">{escape(title)}</text>',
    ]
    for index, line in enumerate(lines):
        parts.append(
            f'<text class="sub" x="{center:g}" y="{first_baseline + 18 + 15 * index:g}">{escape(line)}</text>'
        )
    parts.append("</g>")
    return "".join(parts)


def _graph_svg(viewbox, nodes, edges, labels, texts, label_texts, aria, marker_id, extra="") -> str:
    width, height = viewbox
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(aria)}">',
        f'<defs><marker id="{marker_id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path class="arrow" d="M0 0 L10 5 L0 10 z"/></marker></defs>',
    ]
    parts.extend(f'<path class="edge" d="{d}" marker-end="url(#{marker_id})"/>' for d in edges)
    parts.extend(_node_svg(node, texts[node[0]]) for node in nodes)
    parts.extend(
        f'<text class="edge-label" x="{x}" y="{y}">{escape(label_texts[key])}</text>' for key, x, y in labels
    )
    parts.append(extra)
    parts.append("</svg>")
    return "".join(parts)


def flow_graph(lang: str) -> str:
    text = GRAPH_TEXT[lang]
    return _graph_svg(
        FLOW_VIEWBOX,
        FLOW_NODES,
        FLOW_EDGES,
        FLOW_LABELS,
        text["flow"],
        text["flow_labels"],
        text["aria_flow"],
        "af",
    )


def info_graph(lang: str) -> str:
    text = GRAPH_TEXT[lang]
    x, y = INFO_TITLE_POS
    title = f'<text class="band-title" x="{x}" y="{y}">{escape(text["info_title"])}</text>'
    return _graph_svg(
        INFO_VIEWBOX,
        INFO_NODES,
        INFO_EDGES,
        INFO_LABELS,
        text["info"],
        text["info_labels"],
        text["aria_info"],
        "ai",
        extra=title,
    )


def legend(lang: str) -> str:
    items = "".join(
        f'<li><span class="swatch {kind}"></span>{escape(label)}</li>'
        for kind, label in GRAPH_TEXT[lang]["legend"].items()
    )
    return f'<ul class="legend">{items}</ul>'


CSS = """
:root {
  --bg: #f6f7f9; --panel: #ffffff; --text: #1c2430; --muted: #5b6675; --line: #d9dee5;
  --accent: #1d4ed8; --code-bg: #eef1f5; --warn-bg: #fff6e5; --warn-line: #b7791f;
  --input: #e8eefc; --input-line: #3b5fc4; --control: #fdeee3; --control-line: #bf6a24;
  --endpoint: #e9e8fb; --endpoint-line: #5a50c2; --process: #eef1f5; --process-line: #66717f;
  --store: #e3f4ea; --store-line: #2c8050; --output: #fbf3d6; --output-line: #94790f;
  --muted-node: #f3f4f6; --muted-node-line: #9aa3af;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f141b; --panel: #171e27; --text: #e6eaf0; --muted: #9ba7b6; --line: #2d3743;
    --accent: #8ab4ff; --code-bg: #212a35; --warn-bg: #2c2415; --warn-line: #d39e3e;
    --input: #1c2a4a; --input-line: #7f9ee8; --control: #3a281a; --control-line: #d98e50;
    --endpoint: #2a2750; --endpoint-line: #9a92ea; --process: #252d38; --process-line: #8e99a8;
    --store: #17352a; --store-line: #5cb98a; --output: #39310f; --output-line: #cdb23f;
    --muted-node: #20262f; --muted-node-line: #6d7785;
  }
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; background: var(--bg); color: var(--text); line-height: 1.6; font-size: 16px;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
header { background: var(--panel); border-bottom: 1px solid var(--line); padding: 28px 16px; }
.wrap { max-width: 960px; margin: 0 auto; padding: 0 16px; }
header .wrap { padding: 0; display: flex; gap: 16px; justify-content: space-between; align-items: flex-start; }
h1 { margin: 0 0 6px; font-size: 28px; line-height: 1.2; }
header p { margin: 0; color: var(--muted); }
.lang { flex-shrink: 0; font-weight: 600; }
nav { margin: 24px 0; padding: 16px 20px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
nav ol { margin: 0; padding-left: 20px; columns: 2; column-gap: 32px; }
section { margin: 36px 0; }
h2 { font-size: 22px; margin: 0 0 12px; padding-bottom: 6px; border-bottom: 1px solid var(--line); }
h3 { font-size: 17px; margin: 22px 0 8px; }
a { color: var(--accent); }
code { font-family: ui-monospace, Consolas, Menlo, monospace; font-size: .9em; background: var(--code-bg);
  padding: 1px 5px; border-radius: 4px; overflow-wrap: anywhere; }
pre { background: var(--code-bg); padding: 14px; border-radius: 8px; overflow-x: auto; }
pre code { background: none; padding: 0; overflow-wrap: normal; }
.table { overflow-x: auto; margin: 12px 0; }
table { border-collapse: collapse; width: 100%; background: var(--panel); font-size: 15px; }
th, td { border: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }
th { background: var(--code-bg); }
.note { background: var(--warn-bg); border-left: 4px solid var(--warn-line); padding: 12px 16px;
  border-radius: 4px; margin: 14px 0; }
.graph { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; overflow-x: auto; }
.graph svg { display: block; width: 100%; min-width: 820px; height: auto; }
.node rect { stroke-width: 1.4; }
.node .title { font-size: 14px; font-weight: 650; fill: var(--text); text-anchor: middle; }
.node .sub { font-size: 12px; fill: var(--muted); text-anchor: middle; }
.input rect { fill: var(--input); stroke: var(--input-line); }
.control rect { fill: var(--control); stroke: var(--control-line); }
.endpoint rect { fill: var(--endpoint); stroke: var(--endpoint-line); }
.process rect { fill: var(--process); stroke: var(--process-line); }
.store rect { fill: var(--store); stroke: var(--store-line); }
.output rect { fill: var(--output); stroke: var(--output-line); }
.muted rect { fill: var(--muted-node); stroke: var(--muted-node-line); stroke-dasharray: 5 3; }
.edge { fill: none; stroke: var(--muted); stroke-width: 1.3; }
.arrow { fill: var(--muted); }
.edge-label { font-size: 12px; fill: var(--muted); text-anchor: middle; }
.band-title { font-size: 14px; font-weight: 650; fill: var(--text); }
.legend { list-style: none; display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 10px 0 0; padding: 0;
  font-size: 14px; color: var(--muted); }
.legend li { display: flex; align-items: center; gap: 6px; }
.swatch { width: 14px; height: 14px; border-radius: 3px; border: 1.4px solid; display: inline-block; }
.swatch.input { background: var(--input); border-color: var(--input-line); }
.swatch.control { background: var(--control); border-color: var(--control-line); }
.swatch.endpoint { background: var(--endpoint); border-color: var(--endpoint-line); }
.swatch.process { background: var(--process); border-color: var(--process-line); }
.swatch.store { background: var(--store); border-color: var(--store-line); }
.swatch.output { background: var(--output); border-color: var(--output-line); }
footer { color: var(--muted); font-size: 14px; padding: 24px 0 48px; border-top: 1px solid var(--line); }
@media (max-width: 640px) { nav ol { columns: 1; } h1 { font-size: 24px; } }
"""

BODY_ES = """
<nav aria-label="Contenido"><ol>
<li><a href="#que-es">Qué es</a></li>
<li><a href="#inicio">Primeros pasos</a></li>
<li><a href="#funcionamiento">Cómo funciona</a></li>
<li><a href="#informacion">Qué datos maneja</a></li>
<li><a href="#pricing">DynamicPricing</a></li>
<li><a href="#route">SmartRoute</a></li>
<li><a href="#lead">LeadMatch</a></li>
<li><a href="#datasets">Administrar datasets</a></li>
<li><a href="#limites">Límites y errores</a></li>
<li><a href="#limitaciones">Limitaciones actuales</a></li>
<li><a href="#operacion">Instalación</a></li>
</ol></nav>

<section id="que-es"><h2>1. Qué es</h2>
<p>Una API para empresas con tres productos de optimización. Cada empresa tiene su propia clave y sus datos quedan separados de los de las demás.</p>
<div class="table"><table>
<tr><th>Producto</th><th>Para qué sirve</th><th>Qué hay que mandarle</th></tr>
<tr><td><strong>DynamicPricing</strong></td><td>Recomendar el precio de cada producto</td><td>Costo, precio actual e historial de ventas, en cada pedido</td></tr>
<tr><td><strong>SmartRoute</strong></td><td>Planificar el reparto de una flota</td><td>Depósito, paradas con horario y vehículos, en cada pedido</td></tr>
<tr><td><strong>LeadMatch</strong></td><td>Asignar cada lead al vendedor más conveniente</td><td>Leads y vendedores; opcionalmente, un histórico para entrenar un modelo propio</td></tr>
</table></div>
<p>Los ejemplos usan <code>http://localhost:8000</code>: reemplazalo por la dirección donde esté instalada la API. La referencia completa de cada campo está en <a href="/docs">/docs</a>.</p>
</section>

<section id="inicio"><h2>2. Primeros pasos</h2>
<h3>Registrar la empresa</h3>
<pre><code>curl -X POST http://localhost:8000/api/v1/companies \\
  -H "Content-Type: application/json" \\
  -d '{"name": "Empresa Demo", "contact_email": "demo@empresa.com"}'</code></pre>
<p>La respuesta trae el campo <code>api_key</code>.</p>
<div class="note"><strong>Guardá la clave en ese momento.</strong> Se muestra una sola vez, porque la plataforma guarda únicamente una huella que no permite recuperarla.</div>
<h3>Usar la clave</h3>
<p>Mandala en el header <code>X-API-Key</code> de cada pedido:</p>
<pre><code>curl http://localhost:8000/api/v1/companies/me -H "X-API-Key: TU_CLAVE"</code></pre>
<h3>Cambiar la clave</h3>
<p>Si creés que se filtró, generá otra. La anterior deja de funcionar en el momento, así que actualizá tus sistemas enseguida.</p>
<pre><code>curl -X POST http://localhost:8000/api/v1/companies/me/rotate-key -H "X-API-Key: TU_CLAVE"</code></pre>
<p>Para generar una nueva hace falta la clave vigente. Si la perdiste, tiene que intervenir quien administra la plataforma.</p>
</section>

<section id="funcionamiento"><h2>3. Cómo funciona</h2>
<p>Todo pedido pasa primero por cuatro controles, en este orden. Después sigue por una de las cinco líneas, según la ruta que se llamó.</p>
<div class="graph">{{FLOW}}</div>{{LEGEND}}
<h3>Cómo leer el grafo</h3>
<ul>
<li><strong>Controles (fila superior):</strong> frenan el abuso antes de gastar recursos y comprueban la clave. Si uno falla, el pedido termina ahí con un error (ver <a href="#limites">Límites y errores</a>).</li>
<li><strong>DynamicPricing y SmartRoute:</strong> calculan todo con lo que llega en el pedido y no guardan nada.</li>
<li><strong>Carga de datos y Entrenamiento:</strong> son opcionales y solo sirven a LeadMatch. El archivo subido alimenta a la regresión logística, que produce una versión nueva en el registro de modelos.</li>
<li><strong>LeadMatch:</strong> usa el modelo vigente de la empresa. Si todavía no hay ninguno, puntúa con una fórmula de respaldo y lo avisa con <code>used_default_scoring: true</code>.</li>
</ul>
</section>

<section id="informacion"><h2>4. Qué datos maneja</h2>
<p>La plataforma guarda solo los datos de la empresa y lo necesario para LeadMatch. Lo que se envía a DynamicPricing, a SmartRoute y a <code>/lead-match/match</code> se usa para responder y se descarta.</p>
<div class="graph">{{INFO}}</div>
<h3>Cómo leer el grafo</h3>
<ul>
<li><strong>Empresa:</strong> de la clave se guarda solo la huella (<code>api_key_hash</code>) y sus primeros caracteres (<code>api_key_prefix</code>), que sirven para reconocerla.</li>
<li><strong>Dataset y Archivo CSV:</strong> una empresa puede tener varios datasets. La tabla describe cada uno y el archivo queda en una carpeta propia de la empresa.</li>
<li><strong>Columnas:</strong> el modelo aprende con cuatro columnas. <code>lead_id</code> y <code>rep_id</code> se aceptan en el archivo, pero no se usan para entrenar.</li>
<li><strong>Versión de modelo y Versión vigente:</strong> cada entrenamiento agrega una versión, con sus métricas y el dataset que usó. <code>current.json</code> indica cuál se usa: una versión nueva solo pasa a ser la vigente si no es peor (ver <a href="#lead">LeadMatch</a>).</li>
</ul>
</section>

<section id="pricing"><h2>5. DynamicPricing</h2>
<p>Estima cuánto cambia la demanda de cada producto según el precio y busca el precio más conveniente dentro de tus reglas.</p>
<pre><code>curl -X POST http://localhost:8000/api/v1/dynamic-pricing/price \\
  -H "X-API-Key: TU_CLAVE" -H "Content-Type: application/json" \\
  -d '{
    "skus": [{
      "sku_id": "sku-1", "base_cost": 100, "current_price": 150, "stock": 50,
      "min_margin_pct": 0.15, "max_daily_variation_pct": 0.10,
      "purchase_events": [
        {"price": 140, "quantity_sold": 30}, {"price": 150, "quantity_sold": 22},
        {"price": 160, "quantity_sold": 15}, {"price": 170, "quantity_sold": 9}
      ]
    }],
    "objective": "margin"
  }'</code></pre>
<div class="table"><table>
<tr><th>Campo</th><th>Significado</th></tr>
<tr><td><code>base_cost</code>, <code>current_price</code></td><td>Costo unitario y precio actual. Obligatorios.</td></tr>
<tr><td><code>purchase_events</code></td><td>Ventas anteriores: precio y cantidad. Hacen falta al menos 4 con ventas y 2 precios distintos para estimar la demanda.</td></tr>
<tr><td><code>stock</code></td><td>Unidades disponibles: no se espera vender más que esto. Debe referirse al mismo período que <code>purchase_events</code>. Obligatorio.</td></tr>
<tr><td><code>min_margin_pct</code></td><td>Ganancia mínima <strong>sobre el costo</strong>. Por defecto, 0,15.</td></tr>
<tr><td><code>max_daily_variation_pct</code></td><td>Cuánto puede moverse el precio respecto del actual. Por defecto, 0,05.</td></tr>
<tr><td><code>competitor_price</code>, <code>max_competitor_premium_pct</code></td><td>Precio del competidor y cuánto por encima de él se puede vender. Por defecto, hasta 0,10 más. El margen mínimo y la variación diaria tienen prioridad.</td></tr>
<tr><td><code>objective</code></td><td><code>margin</code> busca la mayor ganancia esperada. <code>volume</code>, la mayor cantidad vendida; si el stock no alcanza, usa el precio más alto con el que igual se vende todo.</td></tr>
</table></div>
<h3>Cómo leer la respuesta</h3>
<ul>
<li><code>recommended_price</code>: el precio sugerido.</li>
<li><code>used_default_elasticity</code>: si es <code>true</code>, no hubo datos para estimar la demanda. Se mantiene el precio actual, ajustado solo si rompe una regla, y <code>expected_quantity</code> y <code>expected_profit</code> vienen vacíos. Sin stock también se mantiene el precio.</li>
<li><code>expected_margin_on_cost_pct</code>: ganancia sobre el costo; es la que se compara con <code>min_margin_pct</code>.</li>
<li><code>expected_margin_pct</code>: ganancia <strong>sobre el precio</strong>. Con costo 100 y precio 115, el primero da 15% y este, 13%.</li>
<li><code>constraints_applied</code>: las reglas aplicadas. Si chocan, gana el margen mínimo, después la variación diaria y por último el competidor, y lo indica acá.</li>
</ul>
</section>

<section id="route"><h2>6. SmartRoute</h2>
<p>Reparte las paradas entre los vehículos y define el orden de cada recorrido.</p>
<pre><code>curl -X POST http://localhost:8000/api/v1/smart-route/plan-routes \\
  -H "X-API-Key: TU_CLAVE" -H "Content-Type: application/json" \\
  -d '{
    "depot": {"lat": -34.6037, "lon": -58.3816},
    "stops": [
      {"stop_id": "s1", "location": {"lat": -34.61, "lon": -58.38}, "demand": 5,
       "time_window_start_min": 0, "time_window_end_min": 240},
      {"stop_id": "s2", "location": {"lat": -34.60, "lon": -58.40}, "demand": 8,
       "time_window_start_min": 0, "time_window_end_min": 240}
    ],
    "vehicles": [{"vehicle_id": "v1", "capacity": 20}]
  }'</code></pre>
<p>Los horarios se expresan en <strong>minutos desde el inicio del turno</strong>. Valores por defecto: 5 minutos por parada, 30 km/h y 480 minutos de duración máxima por ruta, incluido el regreso al depósito.</p>
<h3>Cómo leer la respuesta</h3>
<ul>
<li><code>routes</code>: para cada vehículo, sus paradas en orden, con llegada y salida, distancia, duración hasta volver al depósito y carga.</li>
<li><code>unassigned_stops</code>: las paradas que no entraron en ningún vehículo por capacidad, horario o duración. <strong>Revisalo siempre</strong> y resolvé esas paradas de otra forma.</li>
</ul>
</section>

<section id="lead"><h2>7. LeadMatch</h2>
<p>Estima la probabilidad de que cada lead compre y lo asigna al vendedor que más suma, sin superar la capacidad de cada uno. Funciona desde el primer día; con tu histórico puede puntuar según tus propios resultados.</p>
<h3>Paso 1 (opcional): subir el histórico</h3>
<p>Archivo <code>.csv</code>, <code>.tsv</code> o <code>.txt</code> de hasta 25 MB. Se aceptan los CSV de Excel en español. Las columnas <code>origin</code>, <code>time_on_site_minutes</code> y <code>company_size</code> describen al lead; <code>converted</code> indica si compró (<code>1</code>/<code>0</code>, <code>sí</code>/<code>no</code>, <code>yes</code>/<code>no</code> o <code>true</code>/<code>false</code>).</p>
<pre><code>curl -X POST http://localhost:8000/api/v1/datasets/lead_match \\
  -H "X-API-Key: TU_CLAVE" -F "file=@leads.csv"</code></pre>
<h3>Paso 2 (opcional): entrenar</h3>
<p>Hacen falta al menos 20 filas completas, con casos de <code>0</code> y de <code>1</code>. Usá el <code>id</code> que devolvió el paso anterior.</p>
<pre><code>curl -X POST http://localhost:8000/api/v1/lead-match/train \\
  -H "X-API-Key: TU_CLAVE" -H "Content-Type: application/json" -d '{"dataset_id": 1}'</code></pre>
<p>La respuesta incluye <code>metrics.auc</code>: 0,5 equivale a puntuar al azar, y cuanto más cerca de 1, mejor.</p>
<p><strong>Entrenar no siempre cambia el modelo en uso.</strong> La versión nueva queda guardada, pero solo pasa a ser la vigente si su AUC llega a 0,55 y no es peor que el del modelo actual medido sobre las mismas filas. Mirá <code>promoted</code> y <code>promotion_reason</code> en la respuesta.</p>
<h3>Paso 3: asignar</h3>
<pre><code>curl -X POST http://localhost:8000/api/v1/lead-match/match \\
  -H "X-API-Key: TU_CLAVE" -H "Content-Type: application/json" \\
  -d '{
    "leads": [{"lead_id": "l1", "origin": "paid_ads", "time_on_site_minutes": 22, "company_size": 300}],
    "reps": [{"rep_id": "r1", "capacity": 2}, {"rep_id": "r2", "capacity": 3, "specialties": ["paid_ads"]}]
  }'</code></pre>
<p>Por vendedor podés indicar <code>historical_close_rate</code> (por defecto, 0,5) y <code>specialties</code>, que multiplican el puntaje por <code>specialty_bonus</code> (por defecto, 1,2).</p>
<h3>Cómo leer la respuesta</h3>
<ul>
<li><code>assignments</code>: lead, vendedor, probabilidad de compra y puntaje.</li>
<li><code>unmatched_leads</code>: leads sin vendedor por falta de capacidad.</li>
<li><code>used_default_scoring</code>: <code>true</code> significa que se usó la fórmula de respaldo en lugar de un modelo propio.</li>
</ul>
<h3>Ver las versiones y volver a una anterior</h3>
<p>Para listar las versiones, con sus métricas y cuál está vigente:</p>
<pre><code>curl http://localhost:8000/api/v1/lead-match/models -H "X-API-Key: TU_CLAVE"</code></pre>
<p>Para dejar vigente una versión, ya sea una anterior o una que el entrenamiento no promovió:</p>
<pre><code>curl -X POST http://localhost:8000/api/v1/lead-match/models/{version_id}/promote -H "X-API-Key: TU_CLAVE"</code></pre>
</section>

<section id="datasets"><h2>8. Administrar datasets</h2>
<div class="table"><table>
<tr><th>Acción</th><th>Ruta</th><th>Detalle</th></tr>
<tr><td>Subir</td><td><code>POST /api/v1/datasets/lead_match</code></td><td>Un archivo idéntico a otro ya subido no se duplica.</td></tr>
<tr><td>Listar</td><td><code>GET /api/v1/datasets/lead_match</code></td><td><code>is_active</code> marca el vigente.</td></tr>
<tr><td>Ver las primeras filas</td><td><code>GET /api/v1/datasets/lead_match/{id}/preview</code></td><td>Muestra hasta 20 filas.</td></tr>
<tr><td>Reemplazar</td><td><code>PUT /api/v1/datasets/lead_match/{id}</code></td><td>Corrige el contenido y conserva el mismo <code>id</code>.</td></tr>
<tr><td>Marcar como vigente</td><td><code>POST /api/v1/datasets/lead_match/{id}/activate</code></td><td>Es informativo: para entrenar siempre se indica el <code>dataset_id</code>.</td></tr>
<tr><td>Borrar</td><td><code>DELETE /api/v1/datasets/lead_match/{id}</code></td><td>Borra el archivo y el registro, sin vuelta atrás. No afecta a los modelos ya entrenados.</td></tr>
</table></div>
</section>

<section id="limites"><h2>9. Límites y errores</h2>
<div class="table"><table>
<tr><th>Límite</th><th>Valor</th></tr>
<tr><td>Pedidos por empresa</td><td>120 por minuto; <code>/train</code>, 5; subir o reemplazar datasets, 10</td></tr>
<tr><td>Pedidos por IP</td><td>600 por minuto</td></tr>
<tr><td>Altas y cambios de clave por IP</td><td>10 por hora cada uno</td></tr>
<tr><td>Tamaño</td><td>25 MB por archivo y 10 MB por pedido JSON</td></tr>
<tr><td>DynamicPricing</td><td>500 productos y 5.000 ventas por producto</td></tr>
<tr><td>SmartRoute</td><td>100 paradas y 100 vehículos</td></tr>
<tr><td>LeadMatch</td><td>1.000 leads, 200 vendedores y 2.000.000 de combinaciones de lead por cupo</td></tr>
</table></div>
<div class="table"><table>
<tr><th>Código</th><th>Qué pasó</th><th>Qué hacer</th></tr>
<tr><td>400</td><td>El archivo no es válido o no parece un dataset de leads</td><td>Revisá el formato y las columnas</td></tr>
<tr><td>401</td><td>Falta la clave o no es válida</td><td>Revisá el header <code>X-API-Key</code></td></tr>
<tr><td>404</td><td>El dataset no existe o es de otra empresa</td><td>Revisá el <code>id</code></td></tr>
<tr><td>409</td><td>El email ya está registrado</td><td>Usá la clave existente</td></tr>
<tr><td>413</td><td>El archivo o el pedido es demasiado grande</td><td>Reducilo o dividilo</td></tr>
<tr><td>422</td><td>Faltan datos, tienen un formato incorrecto o el pedido supera un tope</td><td>El detalle de la respuesta indica el campo o la columna</td></tr>
<tr><td>429</td><td>Se superó un límite de pedidos</td><td>Esperá y reintentá</td></tr>
</table></div>
</section>

<section id="limitaciones"><h2>10. Limitaciones actuales</h2>
<p>Conviene conocerlas antes de usar los resultados en una decisión real.</p>
<h3>DynamicPricing</h3>
<ul>
<li>Estima la demanda solo a partir del precio: no considera temporada, promociones ni cambios del mercado. Con pocos eventos, la estimación es débil.</li>
<li>El competidor solo funciona como techo: no hace bajar el precio si ya estás por debajo de ese límite.</li>
</ul>
<h3>SmartRoute</h3>
<ul>
<li>Calcula en línea recta y a velocidad constante, sin calles ni tránsito: <strong>los tiempos reales serán mayores</strong>, así que conviene dejar margen en las ventanas horarias.</li>
<li>Busca una buena solución, no necesariamente la mejor posible. Una parada que figura sin asignar podría entrar con otra distribución de las rutas.</li>
<li>Todas las rutas salen del depósito al inicio del turno y hay un solo depósito.</li>
</ul>
<h3>LeadMatch</h3>
<ul>
<li>El modelo usa solo tres datos del lead: origen, tiempo en el sitio y tamaño de la empresa. No aprende del historial de cada vendedor: la tasa de cierre y las especialidades las informás vos.</li>
<li>Promover una versión no prueba que funcione bien con leads futuros: el AUC se mide con datos históricos.</li>
</ul>
<h3>Cuenta</h3>
<ul>
<li>El alta no verifica el email, y el error 409 permite saber si un email ya está registrado.</li>
</ul>
</section>

<section id="operacion"><h2>11. Instalación</h2>
<p>Para quien instala la plataforma. Requiere Python 3.11.</p>
<pre><code>pip install -r requirements.txt
cp .env.example .env
alembic upgrade head
uvicorn app.main:app</code></pre>
<ul>
<li><strong>Base existente:</strong> hacé un backup y corré <code>alembic upgrade head</code> antes de arrancar la versión nueva. Las claves emitidas siguen funcionando, pero esa migración no se puede revertir.</li>
<li><strong>Detrás de un proxy o PaaS:</strong> definí <code>TRUSTED_PROXY_HOPS</code> (ver <code>.env.example</code>). Si no, todos los clientes comparten los límites por IP.</li>
<li><strong>Archivos:</strong> los datasets y modelos se guardan en <code>storage/</code>. En un PaaS, montá ahí un volumen persistente o se perderán al redesplegar.</li>
<li><strong>Varias instancias:</strong> los contadores de límites están en memoria de cada proceso. Para compartirlos hace falta un almacenamiento común, como Redis.</li>
</ul>
</section>
"""

BODY_EN = """
<nav aria-label="Contents"><ol>
<li><a href="#what">What it is</a></li>
<li><a href="#start">Getting started</a></li>
<li><a href="#operation">How it works</a></li>
<li><a href="#information">What data it handles</a></li>
<li><a href="#pricing">DynamicPricing</a></li>
<li><a href="#route">SmartRoute</a></li>
<li><a href="#lead">LeadMatch</a></li>
<li><a href="#datasets">Managing datasets</a></li>
<li><a href="#limits">Limits and errors</a></li>
<li><a href="#limitations">Current limitations</a></li>
<li><a href="#install">Installation</a></li>
</ol></nav>

<section id="what"><h2>1. What it is</h2>
<p>An API for businesses with three optimization products. Each company has its own key, and its data is kept separate from everyone else's.</p>
<div class="table"><table>
<tr><th>Product</th><th>What it does</th><th>What you send it</th></tr>
<tr><td><strong>DynamicPricing</strong></td><td>Recommends a price for each product</td><td>Cost, current price and sales history, on every request</td></tr>
<tr><td><strong>SmartRoute</strong></td><td>Plans deliveries for a fleet</td><td>Depot, stops with time windows and vehicles, on every request</td></tr>
<tr><td><strong>LeadMatch</strong></td><td>Assigns each lead to the best-suited sales rep</td><td>Leads and sales reps; optionally, a history to train your own model</td></tr>
</table></div>
<p>The examples use <code>http://localhost:8000</code>: replace it with the address where the API is installed. The full reference for every field is at <a href="/docs">/docs</a>.</p>
</section>

<section id="start"><h2>2. Getting started</h2>
<h3>Register your company</h3>
<pre><code>curl -X POST http://localhost:8000/api/v1/companies \\
  -H "Content-Type: application/json" \\
  -d '{"name": "Demo Company", "contact_email": "demo@company.com"}'</code></pre>
<p>The response includes the <code>api_key</code> field.</p>
<div class="note"><strong>Save the key right away.</strong> It is shown only once, because the platform stores just a fingerprint that cannot be turned back into the key.</div>
<h3>Use the key</h3>
<p>Send it in the <code>X-API-Key</code> header of every request:</p>
<pre><code>curl http://localhost:8000/api/v1/companies/me -H "X-API-Key: YOUR_KEY"</code></pre>
<h3>Change the key</h3>
<p>If you think it has leaked, generate another one. The old key stops working immediately, so update your systems straight away.</p>
<pre><code>curl -X POST http://localhost:8000/api/v1/companies/me/rotate-key -H "X-API-Key: YOUR_KEY"</code></pre>
<p>Generating a new key requires the current one. If you have lost it, the platform administrator has to step in.</p>
</section>

<section id="operation"><h2>3. How it works</h2>
<p>Every request first goes through four controls, in this order. It then follows one of the five lines, depending on the route called.</p>
<div class="graph">{{FLOW}}</div>{{LEGEND}}
<h3>How to read the graph</h3>
<ul>
<li><strong>Controls (top row):</strong> they stop abuse before resources are spent and check the key. If one fails, the request ends there with an error (see <a href="#limits">Limits and errors</a>).</li>
<li><strong>DynamicPricing and SmartRoute:</strong> they calculate everything from the request itself and store nothing.</li>
<li><strong>Data upload and Training:</strong> these are optional and only serve LeadMatch. The uploaded file feeds the logistic regression, which produces a new version in the model registry.</li>
<li><strong>LeadMatch:</strong> it uses the company's current model. If there is none yet, it scores with a fallback formula and says so with <code>used_default_scoring: true</code>.</li>
</ul>
</section>

<section id="information"><h2>4. What data it handles</h2>
<p>The platform stores only company data and what LeadMatch needs. Whatever you send to DynamicPricing, SmartRoute and <code>/lead-match/match</code> is used to respond and then discarded.</p>
<div class="graph">{{INFO}}</div>
<h3>How to read the graph</h3>
<ul>
<li><strong>Company:</strong> only the key's fingerprint (<code>api_key_hash</code>) and its first characters (<code>api_key_prefix</code>), which help you recognize it, are stored.</li>
<li><strong>Dataset and CSV file:</strong> a company can have several datasets. The table describes each one, and the file lives in the company's own folder.</li>
<li><strong>Columns:</strong> the model learns from four columns. <code>lead_id</code> and <code>rep_id</code> are accepted in the file but not used for training.</li>
<li><strong>Model version and Current version:</strong> each training run adds a version, with its metrics and the dataset it used. <code>current.json</code> says which one is in use: a new version only becomes current if it is not worse (see <a href="#lead">LeadMatch</a>).</li>
</ul>
</section>

<section id="pricing"><h2>5. DynamicPricing</h2>
<p>It estimates how demand for each product changes with price and looks for the best price within your rules.</p>
<pre><code>curl -X POST http://localhost:8000/api/v1/dynamic-pricing/price \\
  -H "X-API-Key: YOUR_KEY" -H "Content-Type: application/json" \\
  -d '{
    "skus": [{
      "sku_id": "sku-1", "base_cost": 100, "current_price": 150, "stock": 50,
      "min_margin_pct": 0.15, "max_daily_variation_pct": 0.10,
      "purchase_events": [
        {"price": 140, "quantity_sold": 30}, {"price": 150, "quantity_sold": 22},
        {"price": 160, "quantity_sold": 15}, {"price": 170, "quantity_sold": 9}
      ]
    }],
    "objective": "margin"
  }'</code></pre>
<div class="table"><table>
<tr><th>Field</th><th>Meaning</th></tr>
<tr><td><code>base_cost</code>, <code>current_price</code></td><td>Unit cost and current price. Required.</td></tr>
<tr><td><code>purchase_events</code></td><td>Past sales: price and quantity. Estimating demand takes at least 4 events with sales and 2 different prices.</td></tr>
<tr><td><code>stock</code></td><td>Units available: no more than this is expected to sell. It must refer to the same period as <code>purchase_events</code>. Required.</td></tr>
<tr><td><code>min_margin_pct</code></td><td>Minimum profit <strong>on cost</strong>. Default: 0.15.</td></tr>
<tr><td><code>max_daily_variation_pct</code></td><td>How far the price may move from the current one. Default: 0.05.</td></tr>
<tr><td><code>competitor_price</code>, <code>max_competitor_premium_pct</code></td><td>The competitor's price and how far above it you may sell. Default: up to 0.10 more. Minimum margin and daily variation take priority.</td></tr>
<tr><td><code>objective</code></td><td><code>margin</code> seeks the highest expected profit. <code>volume</code>, the most units sold; if stock runs short, it uses the highest price that still sells everything.</td></tr>
</table></div>
<h3>How to read the response</h3>
<ul>
<li><code>recommended_price</code>: the suggested price.</li>
<li><code>used_default_elasticity</code>: if <code>true</code>, there was no data to estimate demand. The current price is kept, adjusted only if it breaks a rule, and <code>expected_quantity</code> and <code>expected_profit</code> come back empty. The price is also kept when there is no stock.</li>
<li><code>expected_margin_on_cost_pct</code>: profit on cost; this is the one to compare with <code>min_margin_pct</code>.</li>
<li><code>expected_margin_pct</code>: profit <strong>on price</strong>. With a cost of 100 and a price of 115, the first is 15% and this one is 13%.</li>
<li><code>constraints_applied</code>: the rules applied. If they conflict, minimum margin wins, then daily variation, and the competitor last, and this field says so.</li>
</ul>
</section>

<section id="route"><h2>6. SmartRoute</h2>
<p>It distributes stops among vehicles and sets the order of each trip.</p>
<pre><code>curl -X POST http://localhost:8000/api/v1/smart-route/plan-routes \\
  -H "X-API-Key: YOUR_KEY" -H "Content-Type: application/json" \\
  -d '{
    "depot": {"lat": -34.6037, "lon": -58.3816},
    "stops": [
      {"stop_id": "s1", "location": {"lat": -34.61, "lon": -58.38}, "demand": 5,
       "time_window_start_min": 0, "time_window_end_min": 240},
      {"stop_id": "s2", "location": {"lat": -34.60, "lon": -58.40}, "demand": 8,
       "time_window_start_min": 0, "time_window_end_min": 240}
    ],
    "vehicles": [{"vehicle_id": "v1", "capacity": 20}]
  }'</code></pre>
<p>Times are expressed in <strong>minutes from the start of the shift</strong>. Defaults: 5 minutes per stop, 30 km/h and a maximum of 480 minutes per route, including the return to the depot.</p>
<h3>How to read the response</h3>
<ul>
<li><code>routes</code>: for each vehicle, its stops in order, with arrival and departure, distance, duration until back at the depot, and load.</li>
<li><code>unassigned_stops</code>: the stops that did not fit in any vehicle because of capacity, schedule or duration. <strong>Always check this</strong> and handle those stops another way.</li>
</ul>
</section>

<section id="lead"><h2>7. LeadMatch</h2>
<p>It estimates how likely each lead is to buy and assigns it to the sales rep who adds the most, without exceeding anyone's capacity. It works from day one; with your history it can score based on your own results.</p>
<h3>Step 1 (optional): upload your history</h3>
<p>A <code>.csv</code>, <code>.tsv</code> or <code>.txt</code> file of up to 25 MB. CSV files from Spanish-language Excel are accepted. The columns <code>origin</code>, <code>time_on_site_minutes</code> and <code>company_size</code> describe the lead; <code>converted</code> says whether the lead bought (<code>1</code>/<code>0</code>, <code>yes</code>/<code>no</code>, <code>sí</code>/<code>no</code> or <code>true</code>/<code>false</code>).</p>
<pre><code>curl -X POST http://localhost:8000/api/v1/datasets/lead_match \\
  -H "X-API-Key: YOUR_KEY" -F "file=@leads.csv"</code></pre>
<h3>Step 2 (optional): train</h3>
<p>This takes at least 20 complete rows, with both <code>0</code> and <code>1</code> cases. Use the <code>id</code> returned by the previous step.</p>
<pre><code>curl -X POST http://localhost:8000/api/v1/lead-match/train \\
  -H "X-API-Key: YOUR_KEY" -H "Content-Type: application/json" -d '{"dataset_id": 1}'</code></pre>
<p>The response includes <code>metrics.auc</code>: 0.5 is the same as scoring at random, and the closer to 1, the better.</p>
<p><strong>Training does not always change the model in use.</strong> The new version is saved, but it only becomes current if its AUC reaches 0.55 and it is no worse than the current model measured on the same rows. Check <code>promoted</code> and <code>promotion_reason</code> in the response.</p>
<h3>Step 3: assign</h3>
<pre><code>curl -X POST http://localhost:8000/api/v1/lead-match/match \\
  -H "X-API-Key: YOUR_KEY" -H "Content-Type: application/json" \\
  -d '{
    "leads": [{"lead_id": "l1", "origin": "paid_ads", "time_on_site_minutes": 22, "company_size": 300}],
    "reps": [{"rep_id": "r1", "capacity": 2}, {"rep_id": "r2", "capacity": 3, "specialties": ["paid_ads"]}]
  }'</code></pre>
<p>For each rep you can set <code>historical_close_rate</code> (default: 0.5) and <code>specialties</code>, which multiply the score by <code>specialty_bonus</code> (default: 1.2).</p>
<h3>How to read the response</h3>
<ul>
<li><code>assignments</code>: lead, sales rep, purchase probability and score.</li>
<li><code>unmatched_leads</code>: leads with no rep because of insufficient capacity.</li>
<li><code>used_default_scoring</code>: <code>true</code> means the fallback formula was used instead of your own model.</li>
</ul>
<h3>View versions and return to an earlier one</h3>
<p>To list the versions, with their metrics and which one is current:</p>
<pre><code>curl http://localhost:8000/api/v1/lead-match/models -H "X-API-Key: YOUR_KEY"</code></pre>
<p>To make a version current, whether an earlier one or one that training did not promote:</p>
<pre><code>curl -X POST http://localhost:8000/api/v1/lead-match/models/{version_id}/promote -H "X-API-Key: YOUR_KEY"</code></pre>
</section>

<section id="datasets"><h2>8. Managing datasets</h2>
<div class="table"><table>
<tr><th>Action</th><th>Route</th><th>Details</th></tr>
<tr><td>Upload</td><td><code>POST /api/v1/datasets/lead_match</code></td><td>A file identical to one already uploaded is not duplicated.</td></tr>
<tr><td>List</td><td><code>GET /api/v1/datasets/lead_match</code></td><td><code>is_active</code> marks the current one.</td></tr>
<tr><td>View first rows</td><td><code>GET /api/v1/datasets/lead_match/{id}/preview</code></td><td>Shows up to 20 rows.</td></tr>
<tr><td>Replace</td><td><code>PUT /api/v1/datasets/lead_match/{id}</code></td><td>Corrects the content and keeps the same <code>id</code>.</td></tr>
<tr><td>Mark as current</td><td><code>POST /api/v1/datasets/lead_match/{id}/activate</code></td><td>For information only: training always takes a <code>dataset_id</code>.</td></tr>
<tr><td>Delete</td><td><code>DELETE /api/v1/datasets/lead_match/{id}</code></td><td>Deletes the file and the record, permanently. Models already trained are not affected.</td></tr>
</table></div>
</section>

<section id="limits"><h2>9. Limits and errors</h2>
<div class="table"><table>
<tr><th>Limit</th><th>Value</th></tr>
<tr><td>Requests per company</td><td>120 per minute; <code>/train</code>, 5; uploading or replacing datasets, 10</td></tr>
<tr><td>Requests per IP</td><td>600 per minute</td></tr>
<tr><td>Sign-ups and key changes per IP</td><td>10 per hour each</td></tr>
<tr><td>Size</td><td>25 MB per file and 10 MB per JSON request</td></tr>
<tr><td>DynamicPricing</td><td>500 products and 5,000 sales per product</td></tr>
<tr><td>SmartRoute</td><td>100 stops and 100 vehicles</td></tr>
<tr><td>LeadMatch</td><td>1,000 leads, 200 sales reps and 2,000,000 lead-by-slot combinations</td></tr>
</table></div>
<div class="table"><table>
<tr><th>Code</th><th>What happened</th><th>What to do</th></tr>
<tr><td>400</td><td>The file is invalid or does not look like a leads dataset</td><td>Check the format and columns</td></tr>
<tr><td>401</td><td>The key is missing or invalid</td><td>Check the <code>X-API-Key</code> header</td></tr>
<tr><td>404</td><td>The dataset does not exist or belongs to another company</td><td>Check the <code>id</code></td></tr>
<tr><td>409</td><td>The email is already registered</td><td>Use the existing key</td></tr>
<tr><td>413</td><td>The file or request is too large</td><td>Reduce or split it</td></tr>
<tr><td>422</td><td>Data is missing, badly formatted, or the request exceeds a cap</td><td>The response detail names the field or column</td></tr>
<tr><td>429</td><td>A request limit was exceeded</td><td>Wait and retry</td></tr>
</table></div>
</section>

<section id="limitations"><h2>10. Current limitations</h2>
<p>These are worth knowing before you use the results in a real decision.</p>
<h3>DynamicPricing</h3>
<ul>
<li>It estimates demand from price alone: it does not consider seasonality, promotions or market changes. With few events, the estimate is weak.</li>
<li>The competitor only acts as a ceiling: it does not push the price down if you are already below that limit.</li>
</ul>
<h3>SmartRoute</h3>
<ul>
<li>It calculates in straight lines at constant speed, with no streets or traffic: <strong>real times will be longer</strong>, so leave some slack in your time windows.</li>
<li>It looks for a good solution, not necessarily the best possible one. A stop listed as unassigned might fit with a different distribution of routes.</li>
<li>All routes leave the depot at the start of the shift, and there is a single depot.</li>
</ul>
<h3>LeadMatch</h3>
<ul>
<li>The model uses only three lead attributes: origin, time on site and company size. It does not learn from each rep's history: you supply the close rate and specialties.</li>
<li>Promoting a version does not prove it will work well on future leads: AUC is measured on historical data.</li>
</ul>
<h3>Account</h3>
<ul>
<li>Sign-up does not verify the email, and the 409 error reveals whether an email is already registered.</li>
</ul>
</section>

<section id="install"><h2>11. Installation</h2>
<p>For whoever installs the platform. Requires Python 3.11.</p>
<pre><code>pip install -r requirements.txt
cp .env.example .env
alembic upgrade head
uvicorn app.main:app</code></pre>
<ul>
<li><strong>Existing database:</strong> back it up and run <code>alembic upgrade head</code> before starting the new version. Keys already issued keep working, but that migration cannot be reverted.</li>
<li><strong>Behind a proxy or PaaS:</strong> set <code>TRUSTED_PROXY_HOPS</code> (see <code>.env.example</code>). Otherwise, all clients share the per-IP limits.</li>
<li><strong>Files:</strong> datasets and models are stored in <code>storage/</code>. On a PaaS, mount a persistent volume there or they will be lost on redeploy.</li>
<li><strong>Multiple instances:</strong> limit counters live in each process's memory. Sharing them requires common storage such as Redis.</li>
</ul>
</section>
"""

PAGES = {
    "es": {
        "file": "manual.html",
        "title": "Manual de la plataforma",
        "heading": "Manual de uso",
        "subtitle": "Plataforma B2B de optimización: DynamicPricing, SmartRoute y LeadMatch",
        "switch_href": "/manual/en",
        "switch_label": "English",
        "footer": 'Referencia técnica de la API: <a href="/docs">/docs</a>.',
        "body": BODY_ES,
    },
    "en": {
        "file": "manual_en.html",
        "title": "Platform manual",
        "heading": "User manual",
        "subtitle": "B2B optimization platform: DynamicPricing, SmartRoute and LeadMatch",
        "switch_href": "/manual",
        "switch_label": "Español",
        "footer": 'Technical API reference: <a href="/docs">/docs</a>.',
        "body": BODY_EN,
    },
}


def render(lang: str) -> str:
    page = PAGES[lang]
    body = (
        page["body"]
        .replace("{{FLOW}}", flow_graph(lang))
        .replace("{{INFO}}", info_graph(lang))
        .replace("{{LEGEND}}", legend(lang))
    )
    return (
        f'<!doctype html>\n<html lang="{lang}">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{page['title']}</title>\n<style>{CSS}</style>\n</head>\n<body>\n"
        f'<header><div class="wrap"><div><h1>{page["heading"]}</h1><p>{page["subtitle"]}</p></div>'
        f'<a class="lang" href="{page["switch_href"]}">{page["switch_label"]}</a></div></header>\n'
        f'<main class="wrap">{body}</main>\n'
        f'<div class="wrap"><footer>{page["footer"]}</footer></div>\n</body>\n</html>\n'
    )


def main() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    for lang, page in PAGES.items():
        (STATIC_DIR / page["file"]).write_text(render(lang), encoding="utf-8", newline="\n")
        print(f"Generado: {STATIC_DIR / page['file']}")


if __name__ == "__main__":
    main()
