"""
Configuración centralizada de logging. Se importa por su efecto secundario (solo para
que `logging.basicConfig` corra una vez) desde `app/main.py` (proceso de la API) y desde
`celery_app.py` (proceso del worker), así ambos loguean con el mismo formato en vez de
depender de `print()` sueltos como antes (ver scripts/train_case*.py).

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
