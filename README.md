# Plataforma de Algoritmos Prescriptivos para Negocios Digitales

Repositorio base (FastAPI) para los 3 casos de estudio del proyecto de investigación
**"Formulación de Algoritmos para Negocios Digitales"** (UMSA — Instituto de Investigación,
Anexo I Formulario de Presentación de Proyectos 2025-2026, Director: Vicente Humberto Monteverde).

El repo implementa la capa de servicio de la arquitectura descrita en cada caso: un backend
FastAPI multi-empresa donde cada empresa cliente sube sus propios datasets, entrena su propia
política (RL) y consulta decisiones prescriptivas en tiempo real.

## Casos cubiertos

| Caso | Nombre | Endpoint principal | Algoritmo |
|---|---|---|---|
| 1.1 | Secuenciación de mensajes multicanal | `POST /api/v1/case1/recommend-next-touchpoint` | DQN / PPO |
| 2.1 | Pricing dinámico con restricciones de equidad | `POST /api/v1/case2/get-price` | PPO + LP (PuLP) + Fairlearn |
| 3.3 | Triaje y priorización de anomalías | `POST /api/v1/case3/alert-webhook` | Clasificador auxiliar (RandomForest) + DQN |
| 6.2 | Asignación de recursos en aceleradora/incubadora | `POST /api/v1/case4/allocate` | PPO + LP (PuLP, knapsack de portfolio) |

Cada caso sigue el mismo patrón interno (`app/cases/<caso>/`):

- `schemas.py` — contratos Pydantic (request/response), fieles al Estado(S)/Acción(A)/Recompensa(R) del documento fuente.
- `env.py` — entorno Gymnasium para entrenamiento offline (no se usa en el request-path). Los 4 casos ya calibran la recompensa simulada con columnas reales del dataset histórico, no solo con el estado observado: tasa de respuesta y etapa del funnel en case1; `demand_index`, `user_segment`, `competitor_avg_price` y una penalización de equidad "en vivo" por segmento en case2 (ver más abajo); la etiqueta real `is_false_positive`/`label` y `estimated_financial_impact` en case3; y en case4, además de tracción/equipo/mercado, `past_response_score`, el apoyo ya recibido el período pasado (con retornos decrecientes) y `runway_months` (bonus de urgencia sobre el capital asignado).
- `policy.py` — carga el modelo `stable-baselines3` entrenado de la empresa; si no existe aún, aplica una **heurística de respaldo** para que el endpoint funcione desde el día 1.
- `service.py` — orquesta clasificador/política/restricciones y arma la respuesta.
- `router.py` — expone los endpoints FastAPI (`/recommend-...`, `/get-price`, `/alert-webhook`, `/train`).

## Multi-tenant: empresas y datasets

1. La empresa se registra: `POST /api/v1/companies` → recibe un `api_key`.
2. Todos los endpoints de negocio requieren el header `X-API-Key`.
3. La empresa sube su dataset histórico (CSV) por caso: `POST /api/v1/datasets/{case}` (ej. `case1_messaging`).
4. Dispara el entrenamiento: `POST /api/v1/case{n}/train` con el `dataset_id` (corre en background, ver `scripts/train_case{n}.py`).
5. El modelo se guarda en `storage/models/<company_id>/<case>/model.zip` y desde ese momento los endpoints de inferencia lo usan automáticamente (antes usan la heurística).

Esto significa que **no hace falta un repo por empresa**: es una sola plataforma, aislada por `company_id` tanto en datos (`storage/datasets/<company_id>/...`) como en modelos.

### Ciclo de vida de un dataset (upload, reemplazo, activación, borrado)

Toda esta lógica vive en `app/services/dataset_ingestion.py` (compartida entre el
router HTTP y `scripts/seed_dataset.py`, ver más abajo), no en el endpoint:

- `POST /api/v1/datasets/{case}` — sube un archivo nuevo. Si su contenido (hash
  sha256) es idéntico al de un dataset ya subido para esa empresa+caso, **no se
  duplica**: se reactiva el existente y se devuelve ese mismo registro. Si el
  contenido es distinto, se crea un dataset nuevo y queda marcado como el
  **activo** para (empresa, caso) — desactivando cualquier otro que lo fuera antes.
- `PUT /api/v1/datasets/{case}/{dataset_id}` — reemplaza el contenido de un dataset
  ya subido, sobre el mismo `dataset_id` (a diferencia de `POST`, que siempre crea
  uno nuevo o reactiva uno existente). Pensado para corregir un upload con datos
  malos sin dejar "huérfano" el `dataset_id` si ya hay un modelo entrenado que lo
  referencia en su metadata (`scripts/train_case{n}.py` graba `dataset_id` y
  `dataset_content_hash` en cada versión, ver `model_registry.py`).
- `POST /api/v1/datasets/{case}/{dataset_id}/activate` — marca un dataset anterior
  como el vigente, sin volver a subir el archivo (por ejemplo, para "volver" a una
  versión de datos previa).
- `DELETE /api/v1/datasets/{case}/{dataset_id}` — borra el archivo y el registro.
  Si era el dataset activo, (empresa, caso) queda sin dataset activo hasta que se
  suba/active otro — no se reactiva ninguno automáticamente.
- `GET /api/v1/datasets/{case}` — lista todos los datasets subidos para ese caso,
  con su campo `is_active`.

`is_active` es solo metadata para que la empresa sepa cuál es "el" dataset vigente
entre varios subidos: **no** cambia qué usa `/train`, que sigue pidiendo
`dataset_id` explícito.

Tanto `POST` como `PUT` validan que el CSV traiga al menos un par de columnas
esperadas para ESE caso (`EXPECTED_COLUMNS` en `dataset_ingestion.py`) antes de
guardar nada: si no trae ninguna, la respuesta es `400` en vez de aceptarlo igual
como "validated" (lo que pasaba antes si, por ejemplo, se subía por error el CSV de
`case2_pricing` al endpoint de `case1_messaging`). El resto del pipeline sigue
siendo tolerante a columnas faltantes puntuales (usa defaults, ver `record_to_*` en
cada `encoding.py`) — esto solo corta el caso de "es claramente el dataset
equivocado". Además, el archivo se guarda en streaming (sin cargar el upload entero
en RAM antes de escribirlo), calculando hash y conteo de filas en la misma pasada.

### Cargar datasets sin pasar por HTTP (desarrollo/tests)

`scripts/seed_dataset.py` inserta un CSV local directo en `storage/datasets/` y la
tabla `datasets`, sin levantar el server ni armar un `curl -F`. Útil para cargar los
CSVs de `synthetic_data/` de una sola vez:

```bash
# Un caso puntual
python -m scripts.seed_dataset --company-id 1 --case case2_pricing --file synthetic_data/case2_pricing.csv

# Los 4 datasets sintéticos de una sola vez
python -m scripts.seed_dataset --company-id 1 --all-synthetic
```

### Datasets sintéticos (`generate_synthetic_data.py`)

Regenera los 4 CSVs de `synthetic_data/` desde cero (`python generate_synthetic_data.py`).
Pensados para poder probar casos límite del negocio sin depender de tener datos reales de
una empresa todavía:

- **case2_pricing**: `min_margin_pct` ya no es fijo (`0.10` para todas las filas) — varía
  por fila (`0.10`/`0.15`/`0.20`/`0.45`/`0.60`, con más peso en el extremo bajo), para poder
  ejercitar el caso límite en el que el margen mínimo exigido supera el multiplicador
  techo (`max_multiplier`) y el filtro de factibilidad LP (`constraints.py`) tiene que
  recortar la recomendación en vez de dejarla pasar siempre sin tocar.
- **case3_anomaly**: la columna de etiqueta se renombró de `label` a `is_false_positive`
  (mismo significado: `1` = falso positivo, `0` = incidente real) para que el nombre diga
  qué es, no un genérico "label". `env.py`, `classifier.py` y `train_case3.py` siguen
  aceptando `label` como alias legado, así que un dataset ya subido por una empresa con el
  nombre viejo no se rompe.
- **case4_accelerator**: el portfolio sintético pasó de 60 a 200 startups, para tener
  variedad suficiente al muestrear portfolios de `MAX_STARTUPS=15` (`encoding.py`) y
  ejercitar mejor la lógica de asignación por fundamentals/apoyo previo/urgencia de runway
  del entorno (ver arriba).

## Estructura del repositorio

```
config.py, database.py, security.py   # config, DB (SQLAlchemy), autenticación por API key
celery_app.py, model_registry.py      # cola de entrenamiento (Celery) y versionado de modelos
app/
  models/        # ORM: Company, Dataset
  schemas/       # Pydantic compartidos
  services/      # dataset_ingestion.py: guardado/hash/activación de datasets (usado por api/ y scripts/)
  api/v1/        # router raíz + endpoints de companies y datasets
  cases/
    case1_messaging/
    case2_pricing/
    case3_anomaly/
    case4_accelerator/
  main.py        # instancia FastAPI, monta todos los routers
scripts/
  train_case1.py train_case2.py train_case3.py train_case4.py   # entrenamiento offline por empresa
  seed_dataset.py                                                # carga de datasets sin pasar por HTTP
tests/
  test_case1.py test_case2.py test_case3.py test_case4.py test_datasets.py test_encoding.py
requirements.txt
Dockerfile / docker-compose.yml
.env.example
```

> Nota: `config.py`, `database.py`, `security.py`, `celery_app.py` y `model_registry.py` viven
> en la raíz del repo (no en `app/core/`, pese a lo que puedan sugerir comentarios viejos).

## Cómo correrlo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
# Docs interactivas: http://localhost:8000/docs
```

O con Docker (incluye Postgres):

```bash
docker compose up --build
```

## Migraciones (Alembic)

El primer `uvicorn app.main:app` corre `init_db()` (`Base.metadata.create_all`), que alcanza
para arrancar en limpio. Pero **no versiona cambios de esquema**: si ya tenés una base con
datos (empresas, datasets, modelos registrados) y modificás un modelo en `app/models/`,
`create_all` no va a alterar la tabla existente. A partir de ahí, usar Alembic:

```bash
# Aplicar todas las migraciones pendientes (incluye la inicial: companies + datasets, y
# la que agrega content_hash/is_active/updated_at a datasets)
alembic upgrade head

# Después de modificar un modelo en app/models/, generar la migración correspondiente
alembic revision --autogenerate -m "descripción del cambio"
# revisar el archivo generado en alembic/versions/ antes de aplicarlo
alembic upgrade head

# Volver atrás un paso
alembic downgrade -1
```

`alembic/env.py` ya está cableado a `settings.DATABASE_URL` (mismo `.env` que usa la API), así
que no hace falta tocar `alembic.ini` para apuntar a otra base.

## Cómo entrenar un caso para una empresa

```bash
# 1) Registrar empresa y guardar el api_key devuelto
curl -X POST localhost:8000/api/v1/companies -H "Content-Type: application/json" \
  -d '{"name": "Empresa Demo", "contact_email": "demo@empresa.com"}'

# 2) Subir dataset histórico
curl -X POST localhost:8000/api/v1/datasets/case2_pricing \
  -H "X-API-Key: <API_KEY>" -F "file=@ventas_historicas.csv"

# 2-bis) Alternativa sin HTTP para desarrollo/tests (ver sección de arriba):
#   python -m scripts.seed_dataset --company-id <ID> --case case2_pricing --file ventas_historicas.csv

# 3) Entrenar (o ejecutar el script directo)
python -m scripts.train_case2 --company-id 1 --dataset-id 1 --algorithm ppo --timesteps 100000

# 4) Consultar precio
curl -X POST localhost:8000/api/v1/case2/get-price -H "X-API-Key: <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"product_id":"sku-1","base_cost":100,"user_segment":"premium","inventory_level_pct":30}'
```

## Fuentes de datos de referencia (por caso, para calibrar/entrenar sin datos internos)

- **Caso 1.1**: exportaciones de HubSpot/Salesforce, datasets de email marketing en Kaggle.
- **Caso 2.1**: [Inside Airbnb](http://insideairbnb.com/get-the-data/), [Kaggle Expedia Personalize](https://www.kaggle.com/c/expedia-personalized-sort), [Fairlearn](https://fairlearn.org/) para las métricas de equidad.
- **Caso 3.3**: [NAB — Numenta Anomaly Benchmark](https://github.com/numenta/NAB), Yahoo Anomaly Detection Dataset.

## Qué falta para producción (pendiente, fuera del alcance del esqueleto)

Esta sección quedó desactualizada en algún momento: varios de los puntos que decía
"pendientes" ya están implementados en el repo (Alembic, la cola real de Celery, el gate
de equidad y el rate limiting por empresa), así que se corrige acá para no confundir a
quien recién llega al proyecto. Lo que realmente falta:

- Integración real con Jira/Opsgenie/Zendesk en `case3_anomaly/service.py`
  (`_create_ticket_if_needed` ya hace el POST al webhook configurado en
  `TICKETING_WEBHOOK_URL` y ya es idempotente por `alert_id` -- ver `app/models/alert_ticket.py`
  -- pero sigue sin haber una integración "de fábrica" con ninguna herramienta puntual).
- Autenticación más robusta para empresas (rotación de API keys; hoy una key es válida
  indefinidamente hasta que se marque `is_active=False` a mano).
- Panel/frontend real para que la empresa suba datasets y vea métricas (`dashboard.html`
  es un panel de pruebas técnico pensado para desarrollo/QA, no una UI para clientes).
- Ya implementado (dejado de referencia, por si se lee esta sección desactualizada en un
  fork o una versión vieja del repo): migraciones con Alembic (`alembic/`), cola real de
  entrenamiento con Celery (`celery_app.py`, `app/tasks.py`), gate de equidad con
  Fairlearn antes de promover un modelo de pricing (`case2_pricing/fairness.py` +
  `scripts/train_case2.py`), y rate limiting por empresa (`rate_limit.py`).
