"""
Tareas de Celery para el (re)entrenamiento de modelos (ver celery_app.py).

Cada tarea delega en el `train_for_company` del caso correspondiente (scripts/train_caseN.py),
que ya se encarga de versionar el modelo resultante en el registro (model_registry.py)
y, en case2, de aplicar el filtro de equidad antes de promoverlo. No hace falta invalidar
ningún caché acá: los `service.py` de cada caso detectan la nueva versión promovida
comparando el version_id vigente en cada request (ver `model_registry.get_current_version_id`),
lo cual funciona incluso corriendo el worker en un proceso/contenedor separado del de la API.
"""

from celery_app import celery_app


@celery_app.task(name="train_case1", bind=True, max_retries=1)
def train_case1_task(self, company_id: int, dataset_id: int, algorithm: str = "dqn", timesteps: int = 50_000):
    from scripts.train_case1 import train_for_company

    return train_for_company(company_id, dataset_id, algorithm, timesteps)


@celery_app.task(name="train_case2", bind=True, max_retries=1)
def train_case2_task(
    self,
    company_id: int,
    dataset_id: int,
    algorithm: str = "ppo",
    timesteps: int = 50_000,
    fairness_penalty_weight: float = 1.0,
):
    from scripts.train_case2 import train_for_company

    return train_for_company(company_id, dataset_id, algorithm, timesteps, fairness_penalty_weight)


@celery_app.task(name="train_case3", bind=True, max_retries=1)
def train_case3_task(self, company_id: int, dataset_id: int, timesteps: int = 50_000):
    from scripts.train_case3 import train_for_company

    return train_for_company(company_id, dataset_id, timesteps)


@celery_app.task(name="train_case4", bind=True, max_retries=1)
def train_case4_task(self, company_id: int, dataset_id: int, algorithm: str = "ppo", timesteps: int = 50_000):
    from scripts.train_case4 import train_for_company

    return train_for_company(company_id, dataset_id, algorithm, timesteps)
