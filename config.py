"""
Configuración centralizada del proyecto.
Lee variables de entorno (ver .env.example) usando pydantic-settings.
"""

from functools import lru_cache

from pydantic import field_validator
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

    @field_validator("DATABASE_URL")
    @classmethod
    def _normalizar_esquema_postgres(cls, v: str) -> str:
        # Algunos proveedores (Heroku, a veces Railway) entregan "postgres://...", que
        # SQLAlchemy 2.x ya no acepta como alias de "postgresql://".
        if v.startswith("postgres://"):
            return "postgresql://" + v[len("postgres://") :]
        return v

    # Carpeta donde se guardan los datasets subidos por cada empresa
    DATASETS_DIR: str = "./storage/datasets"

    # Carpeta donde se guardan los modelos entrenados y versionados (hoy solo LeadMatch, uno por empresa)
    MODELS_DIR: str = "./storage/models"

    CORS_ORIGINS: list[str] = ["*"]

    # No hay cola de tareas ni broker: el scoring de LeadMatch entrena sincrónico en
    # el propio request de /train (ver app/cases/lead_match/train.py), y
    # DynamicPricing/SmartRoute son stateless por request, sin entrenamiento.


@lru_cache
def get_settings() -> Settings:
    return Settings()
