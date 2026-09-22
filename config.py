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

    # Seguridad: la autenticación de empresas es por API Key (ver security.py), no por
    # JWT/sesión firmada, así que no hay un SECRET_KEY de firma en uso -- se quitó el que
    # había acá (con un default inseguro, "changeme-in-.env") porque no lo leía ningún
    # módulo del proyecto y solo generaba una falsa sensación de configuración pendiente.
    API_KEY_HEADER: str = "X-API-Key"

    # Carpeta donde se guardan los datasets subidos por cada empresa
    DATASETS_DIR: str = "./storage/datasets"

    # Carpeta donde se guardan los modelos RL entrenados (uno por empresa/caso)
    MODELS_DIR: str = "./storage/models"

    CORS_ORIGINS: list[str] = ["*"]

    # AGREGADO 2026-09-22: se sacó REDIS_URL (Celery ya no se usa -- ver celery_app.py
    # en el historial de git) y los settings específicos de los 4 casos viejos
    # (FAIRNESS_DP_THRESHOLD de case2, TICKETING_WEBHOOK_URL/TIMEOUT_S de case3).
    # Ninguno de los 3 productos actuales necesita una cola de entrenamiento: el
    # scoring de LeadMatch entrena sincrónico en el propio request de /train (ver
    # app/cases/lead_match/train.py), y DynamicPricing/SmartRoute son stateless
    # por request, sin entrenamiento en absoluto.


@lru_cache
def get_settings() -> Settings:
    return Settings()
