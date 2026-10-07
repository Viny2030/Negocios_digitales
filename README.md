# Plataforma B2B de Optimización y ML para Negocios Digitales

Backend (FastAPI) multi-empresa que expone 3 productos de optimización/ML como API:
cada empresa cliente se registra, recibe un API key y consulta decisiones sobre sus
propios datos, aislada del resto de las empresas en la misma plataforma.

> AGREGADO 2026-09-22: este repo pivotó de una plataforma de 4 casos de estudio
> académicos basados en Reinforcement Learning a estos 3 productos, reutilizando la
> misma infraestructura multi-tenant ya validada (auth por API key, aislamiento por
> empresa, rate limiting, versionado de modelos, Alembic, Docker/CI). Si ves
> referencias a `case1`..`case4` en un fork o una copia vieja del código, son de la
> versión anterior.

## Productos

| Producto | Endpoint principal | Algoritmo | ¿Entrena algo? |
|---|---|---|---|
| **DynamicPricing API** | `POST /api/v1/dynamic-pricing/price` | Regresión log-log (elasticidad precio-demanda) + `scipy.optimize.minimize_scalar` para el precio que maximiza margen o volumen, con margen mínimo y variación diaria máxima como restricciones | No — stateless por request |
| **SmartRoute Fleet** | `POST /api/v1/smart-route/plan-routes` | VRPTW: heurística de ahorros de Clarke-Wright + mejora local 2-opt, con asignación de vehículos por capacidad y reparación de factibilidad de ventanas horarias | No — stateless por request |
| **LeadMatch Engine** | `POST /api/v1/lead-match/match` (+ `POST /api/v1/lead-match/train`) | Regresión logística (`scikit-learn`) para probabilidad de conversión + algoritmo húngaro (`scipy.optimize.linear_sum_assignment`) para la asignación óptima lead-vendedor respetando capacidad | Sí — un clasificador por empresa, entrenado sincrónico (segundos) a partir de un dataset subido |

DynamicPricing y SmartRoute resuelven un problema de optimización distinto en cada
request, con los datos que el cliente manda en el body — no hay estado ni modelo que
versionar. LeadMatch es el único de los 3 con un modelo entrenable por empresa (por
eso es también el único que usa el endpoint de datasets, ver abajo); mientras una
empresa no entrenó nada todavía, el scoring cae a una heurística de respaldo para que
el endpoint funcione desde el día 1 (`used_default_scoring: true` en la respuesta).

Cada producto sigue el mismo patrón interno (`app/cases/<producto>/`):

- `schemas.py` — contratos Pydantic (request/response).
- `service.py` — orquesta la lógica del caso (llama a `optimizer.py`/`solver.py`/
  `matching.py` según el producto) y arma la respuesta.
- `router.py` — expone los endpoints FastAPI, todos detrás de `Depends(get_current_company)`.

Además cada uno tiene sus propios módulos de dominio:

- `dynamic_pricing/elasticity.py` (ajuste de elasticidad) y `optimizer.py` (búsqueda
  del precio óptimo dentro del rango factible).
- `smart_route/solver.py` (Clarke-Wright + 2-opt + asignación de vehículos + reparación
  de factibilidad — el módulo más grande de los 3).
- `lead_match/scoring.py` (wrapper del modelo + heurística de respaldo), `train.py`
  (entrenamiento del pipeline de scikit-learn) y `matching.py` (matching húngaro con
  capacidad y afinidad por especialidad).

## Multi-tenant: empresas y datasets

1. La empresa se registra: `POST /api/v1/companies` → recibe un `api_key`.
2. Todos los endpoints de negocio requieren el header `X-API-Key`.
3. Solo LeadMatch usa datasets: la empresa sube su histórico de leads (CSV) por
   `POST /api/v1/datasets/lead_match`.
4. Dispara el entrenamiento: `POST /api/v1/lead-match/train` con el `dataset_id`
   (corre sincrónico, responde en la misma request — no hace falta cola de tareas).
5. El modelo se guarda versionado en `storage/models/<company_id>/lead_match/versions/`
   con un puntero `current.json`; desde ese momento `/lead-match/match` lo usa
   automáticamente para esa empresa (antes, usa la heurística).

Esto significa que **no hace falta un repo por empresa**: es una sola plataforma,
aislada por `company_id` tanto en datos (`storage/datasets/<company_id>/...`) como en
modelos (`storage/models/<company_id>/...`).

### Ciclo de vida de un dataset (upload, reemplazo, activación, borrado)

Toda esta lógica vive en `app/services/dataset_ingestion.py` (compartida entre el
router HTTP y `scripts/seed_dataset.py`, ver más abajo), no en el endpoint. Aplica
hoy solo al caso `lead_match` (ver `CaseType` en `app/models/dataset.py`):

- `POST /api/v1/datasets/lead_match` — sube un archivo nuevo. Si su contenido (hash
  sha256) es idéntico al de un dataset ya subido para esa empresa, **no se duplica**:
  se reactiva el existente y se devuelve ese mismo registro. Si el contenido es
  distinto, se crea un dataset nuevo y queda marcado como el **activo** —
  desactivando cualquier otro que lo fuera antes.
- `PUT /api/v1/datasets/lead_match/{dataset_id}` — reemplaza el contenido de un
  dataset ya subido, sobre el mismo `dataset_id` (a diferencia de `POST`, que siempre
  crea uno nuevo o reactiva uno existente). Pensado para corregir un upload con datos
  malos sin dejar "huérfano" el `dataset_id` si ya hay un modelo entrenado que lo
  referencia en su metadata.
- `POST /api/v1/datasets/lead_match/{dataset_id}/activate` — marca un dataset
  anterior como el vigente, sin volver a subir el archivo.
- `DELETE /api/v1/datasets/lead_match/{dataset_id}` — borra el archivo y el
  registro. Si era el dataset activo, la empresa queda sin dataset activo hasta que
  se suba/active otro — no se reactiva ninguno automáticamente.
- `GET /api/v1/datasets/lead_match` — lista todos los datasets subidos por esa
  empresa, con su campo `is_active`.
- `GET /api/v1/datasets/lead_match/{dataset_id}/preview` — primeras filas del CSV,
  útil para validar el mapeo de columnas.

`is_active` es solo metadata para que la empresa sepa cuál es "el" dataset vigente
entre varios subidos: **no** cambia qué usa `/train`, que sigue pidiendo `dataset_id`
explícito.

Se aceptan archivos `.csv`, `.tsv` o `.txt` (sin importar mayúsculas), separados por
`,`, `;` o tabulación (se detecta solo, así que un CSV exportado por Excel en español
funciona tal cual), en UTF-8 (con o sin BOM) o latin-1/cp1252.

Tanto `POST` como `PUT` validan que el CSV traiga al menos 2 de las columnas
esperadas (`EXPECTED_COLUMNS` en `dataset_ingestion.py`: `lead_id`, `origin`,
`time_on_site_minutes`, `company_size`, `rep_id`, `converted`) antes de guardar nada
— si no trae ninguna coincidencia, la respuesta es `400` en vez de aceptarlo igual
como válido. Además, el archivo se guarda en streaming (sin cargar el upload entero
en RAM antes de escribirlo), calculando hash y conteo de filas en la misma pasada.

### Cargar datasets sin pasar por HTTP (desarrollo/tests)

`scripts/seed_dataset.py` inserta un CSV local directo en `storage/datasets/` y la
tabla `datasets`, sin levantar el server ni armar un `curl -F`:

```bash
python -m scripts.seed_dataset --company-id 1 --case lead_match --file synthetic_data/lead_match.csv
# o, equivalente, con el flag genérico:
python -m scripts.seed_dataset --company-id 1 --all-synthetic
```

### Dataset sintético (`generate_synthetic_data.py`)

Regenera `synthetic_data/lead_match.csv` desde cero (`python generate_synthetic_data.py`):
600 leads con `origin`, `time_on_site_minutes`, `company_size`, `rep_id` y una columna
`converted` derivada de un score ponderado (tiempo en sitio + tamaño de empresa + sesgo
por origen + ruido, umbralizado en la mediana) para que tenga señal real que aprender —
`tests/test_lead_match.py` verifica que el modelo entrenado con esto supera a la
heurística de respaldo.

DynamicPricing y SmartRoute no tienen dataset propio: sus requests llevan los datos
necesarios inline (`purchase_events` por SKU en DynamicPricing; `stops`/`vehicles` en
SmartRoute) — no hay nada que subir de antemano para poder llamarlos.

## Estructura del repositorio

```
config.py, database.py, security.py   # config, DB (SQLAlchemy), autenticación por API key
model_registry.py                      # versionado de modelos (usado hoy solo por lead_match)
rate_limit.py                          # rate limiting por API key (slowapi)
app/
  models/        # ORM: Company, Dataset
  schemas/       # Pydantic compartidos (CompanyCreate/Out, DatasetOut)
  services/      # dataset_ingestion.py: guardado/hash/activación de datasets
  api/v1/        # router raíz + endpoints de companies y datasets
  cases/
    dynamic_pricing/   # schemas.py, elasticity.py, optimizer.py, service.py, router.py
    smart_route/       # schemas.py, solver.py, service.py, router.py
    lead_match/         # schemas.py, scoring.py, train.py, matching.py, service.py, router.py
  main.py        # instancia FastAPI, monta todos los routers
scripts/
  seed_dataset.py                       # carga de datasets sin pasar por HTTP
tests/
  test_dynamic_pricing.py test_smart_route.py test_lead_match.py test_datasets.py
alembic/         # migraciones (incluye la que retira el esquema de los 4 casos viejos)
dashboard.html   # panel de pruebas técnico (no es UI para clientes, ver abajo)
requirements.txt
Dockerfile / docker-compose.yml
.env.example
```

> Nota: `config.py`, `database.py`, `security.py` y `model_registry.py` viven en la
> raíz del repo (no en `app/core/`, pese a lo que puedan sugerir comentarios viejos).

## Cómo correrlo

Linux/Mac (bash):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
uvicorn app.main:app --reload
# Docs interactivas: http://localhost:8000/docs
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # si falla por ExecutionPolicy: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt -r requirements-dev.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
# Tests y lint
python -m pytest tests -q
ruff check .
```

O con Docker (incluye Postgres):

```bash
docker compose up --build
```

## Migraciones (Alembic)

El primer `uvicorn app.main:app` corre `init_db()` (`Base.metadata.create_all`), que
alcanza para arrancar en limpio. Pero **no versiona cambios de esquema**: si ya tenés
una base con datos y modificás un modelo en `app/models/`, `create_all` no va a
alterar la tabla existente. A partir de ahí, usar Alembic:

```bash
# Aplicar todas las migraciones pendientes (incluye la que retira alert_tickets y
# reduce el enum CaseType a solo "lead_match", ver alembic/versions/8324e3a86121_*.py)
alembic upgrade head

# Después de modificar un modelo en app/models/, generar la migración correspondiente
alembic revision --autogenerate -m "descripción del cambio"
# revisar el archivo generado en alembic/versions/ antes de aplicarlo
alembic upgrade head

# Volver atrás un paso
alembic downgrade -1
```

> **Base creada por `uvicorn` antes de usar Alembic:** si `prescriptive.db` (o el
> Postgres) ya tiene las tablas porque las creó `init_db()`, `alembic upgrade head`
> falla con "table already exists" -- esa base no tiene la tabla `alembic_version`.
> Marcala una sola vez como al día con `alembic stamp head` y desde ahí usá
> `upgrade` normalmente.

`alembic/env.py` ya está cableado a `settings.DATABASE_URL` (mismo `.env` que usa la
API), así que no hace falta tocar `alembic.ini` para apuntar a otra base.

> La migración del pivot (`8324e3a86121`) borra la tabla `alert_tickets` (del caso de
> anomalías viejo) y, solo en Postgres, achica el enum `casetype` a un único valor
> (`lead_match`) — lo que implica borrar cualquier dataset que hubiera quedado
> cargado para los 4 casos viejos. En SQLite (desarrollo/tests) no hace falta migrar
> el enum: `create_all` ya arma la tabla con el valor correcto en cada corrida limpia.

## Ejemplo end-to-end por producto

```bash
# 0) Registrar empresa y guardar el api_key devuelto
curl -X POST localhost:8000/api/v1/companies -H "Content-Type: application/json" \
  -d '{"name": "Empresa Demo", "contact_email": "demo@empresa.com"}'
```

**DynamicPricing** (sin dataset previo, todo inline):

```bash
curl -X POST localhost:8000/api/v1/dynamic-pricing/price -H "X-API-Key: <API_KEY>" \
  -H "Content-Type: application/json" -d '{
    "skus": [{
      "sku_id": "sku-1", "base_cost": 100, "current_price": 150, "stock": 50,
      "min_margin_pct": 0.15, "max_daily_variation_pct": 0.10,
      "purchase_events": [
        {"price": 140, "quantity_sold": 30}, {"price": 150, "quantity_sold": 22},
        {"price": 160, "quantity_sold": 15}, {"price": 170, "quantity_sold": 9}
      ]
    }],
    "objective": "margin"
  }'
```

**SmartRoute** (sin dataset previo, todo inline):

```bash
curl -X POST localhost:8000/api/v1/smart-route/plan-routes -H "X-API-Key: <API_KEY>" \
  -H "Content-Type: application/json" -d '{
    "depot": {"lat": -34.6037, "lon": -58.3816},
    "stops": [
      {"stop_id": "s1", "location": {"lat": -34.61, "lon": -58.38}, "demand": 5,
       "time_window_start_min": 0, "time_window_end_min": 240},
      {"stop_id": "s2", "location": {"lat": -34.60, "lon": -58.40}, "demand": 8,
       "time_window_start_min": 0, "time_window_end_min": 240}
    ],
    "vehicles": [{"vehicle_id": "v1", "capacity": 20}]
  }'
```

**LeadMatch** (requiere dataset + entrenamiento primero):

```bash
# 1) Subir el histórico de leads
curl -X POST localhost:8000/api/v1/datasets/lead_match \
  -H "X-API-Key: <API_KEY>" -F "file=@synthetic_data/lead_match.csv"

# 1-bis) Alternativa sin HTTP para desarrollo/tests (ver sección de arriba):
#   python -m scripts.seed_dataset --company-id <ID> --case lead_match --file synthetic_data/lead_match.csv

# 2) Entrenar el clasificador de conversión de esta empresa
curl -X POST localhost:8000/api/v1/lead-match/train -H "X-API-Key: <API_KEY>" \
  -H "Content-Type: application/json" -d '{"dataset_id": 1}'

# 3) Pedir el matching lead → vendedor
curl -X POST localhost:8000/api/v1/lead-match/match -H "X-API-Key: <API_KEY>" \
  -H "Content-Type: application/json" -d '{
    "leads": [{"lead_id": "l1", "origin": "paid_ads", "time_on_site_minutes": 22, "company_size": 300}],
    "reps": [{"rep_id": "r1", "capacity": 2}, {"rep_id": "r2", "capacity": 3, "specialties": ["paid_ads"]}]
  }'
```

### Lo mismo desde PowerShell

En Windows PowerShell 5, `curl` es un alias de `Invoke-WebRequest` y no acepta los
flags de arriba. Usá `Invoke-RestMethod` (o `curl.exe` explícito):

```powershell
$B = "http://localhost:8000/api/v1"
$co = Invoke-RestMethod -Method Post "$B/companies" -ContentType "application/json" `
  -Body '{"name": "Empresa Demo", "contact_email": "demo@empresa.com"}'
$H = @{ "X-API-Key" = $co.api_key }

# DynamicPricing
$body = @{
  skus = @(@{ sku_id = "sku-1"; base_cost = 100; current_price = 150; stock = 50
    purchase_events = @(@{price=140;quantity_sold=30}, @{price=150;quantity_sold=22},
                        @{price=160;quantity_sold=15}, @{price=170;quantity_sold=9}) })
  objective = "margin"
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post "$B/dynamic-pricing/price" -Headers $H -ContentType "application/json" -Body $body

# LeadMatch: subir dataset (curl.exe, porque Invoke-RestMethod -Form requiere PowerShell 7)
curl.exe -X POST "$B/datasets/lead_match" -H "X-API-Key: $($co.api_key)" -F "file=@synthetic_data/lead_match.csv"
Invoke-RestMethod -Method Post "$B/lead-match/train" -Headers $H -ContentType "application/json" -Body '{"dataset_id": 1}'
```

## Panel de pruebas (`dashboard.html`)

Página estática, sin build, pensada para desarrollo/QA técnico (no es una UI para
clientes finales). Lee `/openapi.json` al conectar y arma los formularios de cada
producto automáticamente a partir del schema real de cada endpoint, así que no se
desactualiza si cambia un campo de un `schemas.py`. Para usarla: levantar la API
(`uvicorn app.main:app --reload`), abrir `dashboard.html` (servido por un `http.server`
si el `fetch` falla por CORS al abrirlo con `file://`), pegar un API key y tocar
"Conectar y cargar API".

## Fuentes de datos de referencia (para probar sin datos internos)

- **DynamicPricing**: cualquier histórico de ventas con precio y unidades vendidas por
  fecha/SKU sirve para simular `purchase_events` — por ejemplo exports de e-commerce o
  el dataset de [Kaggle Expedia Personalize](https://www.kaggle.com/c/expedia-personalized-sort)
  reagrupado por producto y precio.
- **SmartRoute**: cualquier dataset de VRPTW público (por ejemplo las instancias
  clásicas de Solomon) sirve para probar el solver con volúmenes más grandes que los
  ejemplos sintéticos de los tests.
- **LeadMatch**: exportaciones de CRM (HubSpot/Salesforce) con origen del lead, tiempo
  en sitio y resultado de conversión.

## Qué falta para producción

- **Autenticación más robusta para empresas**: hoy un `api_key` es válido
  indefinidamente hasta que se marque `is_active=False` a mano — no hay rotación.
- **Panel/frontend real para clientes**: `dashboard.html` es un panel de pruebas
  técnico para desarrollo/QA, no una UI pensada para que una empresa cliente use día a
  día.
- **Persistencia/versionado de resultados de SmartRoute y DynamicPricing**: al ser
  stateless por diseño, hoy no queda un historial consultable de qué precio o ruta se
  recomendó en el pasado — si hiciera falta auditar recomendaciones, habría que
  agregar un registro explícito (no forma parte del alcance actual).
- Ya implementado (dejado de referencia): migraciones con Alembic (`alembic/`),
  versionado de modelos por empresa con rollback (`model_registry.py`, usado por
  LeadMatch), rate limiting por empresa (`rate_limit.py`), y validación de esquema de
  CSV al subir un dataset (`dataset_ingestion.py`).
