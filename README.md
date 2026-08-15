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
| 2.1 | Pricing dinámico con restricciones de equidad | `POST /api/v1/case2/get-price` | PPO/DQN + LP (PuLP) + Fairlearn |
| 3.3 | Triaje y priorización de anomalías | `POST /api/v1/case3/alert-webhook` | Clasificador auxiliar (RandomForest) + DQN |
| 6.2 | Asignación de recursos en aceleradora/incubadora | `POST /api/v1/case4/allocate` | PPO + LP (PuLP, knapsack de portfolio) |

Cada caso sigue el mismo patrón interno (`app/cases/<caso>/`):

- `schemas.py` — contratos Pydantic (request/response), fieles al Estado(S)/Acción(A)/Recompensa(R) del documento fuente.
- `env.py` — entorno Gymnasium para entrenamiento offline (no se usa en el request-path).
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

## Estructura del repositorio

```
config.py, database.py, security.py   # config, DB (SQLAlchemy), autenticación por API key
celery_app.py, model_registry.py      # cola de entrenamiento (Celery) y versionado de modelos
app/
  models/        # ORM: Company, Dataset
  schemas/       # Pydantic compartidos
  api/v1/        # router raíz + endpoints de companies y datasets
  cases/
    case1_messaging/
    case2_pricing/
    case3_anomaly/
    case4_accelerator/
  main.py        # instancia FastAPI, monta todos los routers
scripts/
  train_case1.py train_case2.py train_case3.py train_case4.py   # entrenamiento offline por empresa
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
# Aplicar todas las migraciones pendientes (incluye la inicial: companies + datasets)
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

- Migraciones con Alembic en vez de `create_all` en el startup.
- Cola real de entrenamiento (Celery/RQ) en vez de `BackgroundTasks` de FastAPI.
- Integración real con Jira/Opsgenie/Zendesk en `case3_anomaly/service.py` (`_create_ticket_if_needed`).
- Evaluación automática de Demographic Parity (Fairlearn) como gate antes de promover un modelo de pricing a producción.
- Autenticación más robusta (rotación de API keys, rate limiting por empresa).
- Panel/frontend para que la empresa suba datasets y vea métricas (fuera del alcance de este repo backend).
