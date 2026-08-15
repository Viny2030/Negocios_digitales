"""
Configuración centralizada del proyecto.
Lee variables de entorno (ver .env.example) usando pydantic-settings.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PROJECT_NAME: str = "Plataforma de Algoritmos Prescriptivos - Negocios Digitales"
    ENV: str = "development"

    # Base de datos (Postgres en producción, SQLite para desarrollo local)
    DATABASE_URL: str = "sqlite:///./prescriptive.db"

    # Seguridad
    SECRET_KEY: str = "changeme-in-.env"
    API_KEY_HEADER: str = "X-API-Key"

    # Carpeta donde se guardan los datasets subidos por cada empresa
    DATASETS_DIR: str = "./storage/datasets"

    # Carpeta donde se guardan los modelos RL entrenados (uno por empresa/caso)
    MODELS_DIR: str = "./storage/models"

    CORS_ORIGINS: list[str] = ["*"]

    # Broker/backend de Celery (cola real de entrenamiento, ver celery_app.py).
    # En docker-compose apunta al servicio `redis`; en local, requiere un Redis corriendo
    # (Docker, WSL, o Memurai en Windows) en este mismo host y puerto.
    REDIS_URL: str = "redis://localhost:6379/0"

    # Gating de equidad (case2 - pricing): si la diferencia de Demographic Parity entre
    # segmentos supera este umbral tras entrenar, el modelo se guarda como versión pero
    # NO se promueve a producción (sigue sirviendo la versión anterior aprobada).
    FAIRNESS_DP_THRESHOLD: float = 0.10

    # Ticketing (case3 - anomaly triage): URL de un webhook genérico (Jira/Opsgenie/Zendesk
    # automation webhook, o un endpoint intermedio propio) al que se le hace POST con el ticket
    # a crear. Si queda vacío, no se crea ningún ticket real y solo se deja constancia en el log
    # (útil en desarrollo/demo, donde la empresa todavía no configuró su integración).
    TICKETING_WEBHOOK_URL: str = ""
    TICKETING_WEBHOOK_TIMEOUT_S: float = 5.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
