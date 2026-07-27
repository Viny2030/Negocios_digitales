"""
App de Celery: cola de tareas real para el (re)entrenamiento de modelos, en reemplazo de
`fastapi.BackgroundTasks` (que corría el entrenamiento en el mismo proceso que atiende
requests HTTP, compitiendo por CPU con el resto de la API y sin sobrevivir a un restart
ni reintentar si fallaba).

Requiere un broker Redis (ver `docker-compose.yml`, servicio `redis`, o un Redis local).

Levantar un worker:
    celery -A app.core.celery_app worker --loglevel=info --pool=solo   # Windows
    celery -A app.core.celery_app worker --loglevel=info               # Linux/Mac/Docker

`--pool=solo` es necesario en Windows porque Celery no soporta el pool `prefork`
(basado en os.fork) fuera de Linux/Mac; en el contenedor Docker (Linux) se usa el pool
por defecto.
"""
from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "negocios_digitales",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    # Un entrenamiento colgado no debe bloquear el worker para siempre.
    task_time_limit=60 * 60,
)
