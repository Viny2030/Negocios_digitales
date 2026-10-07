"""
Configuración centralizada de logging. Se importa por su efecto secundario (solo para
que `logging.basicConfig` corra una vez) desde `app/main.py`, así toda la app loguea
con el mismo formato.

Nivel configurable con la variable de entorno LOG_LEVEL (INFO por defecto).
"""

import logging
import os

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
