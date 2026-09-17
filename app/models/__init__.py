"""
Importa todos los modelos ORM acá para que cualquier import de `app.models.*`
registre ambas clases en el mismo registry de SQLAlchemy. Sin esto, scripts que
solo hacen `from app.models.dataset import Dataset` (sin pasar por database.init_db())
fallan al resolver la relationship("Company") porque esa clase nunca se importó.
"""

from app.models.alert_ticket import AlertTicket  # noqa: F401
from app.models.company import Company  # noqa: F401
from app.models.dataset import Dataset  # noqa: F401
