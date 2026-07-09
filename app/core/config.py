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


@lru_cache
def get_settings() -> Settings:
    return Settings()
