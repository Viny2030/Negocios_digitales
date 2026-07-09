"""
Fixtures compartidas de test.

IMPORTANTE: seteamos DATABASE_URL a un archivo en /tmp ANTES de importar app.main, porque
app.core.database crea el engine de produccion una sola vez al importarse (a nivel de modulo).
Si se deja el default relativo ("./prescriptive.db"), sqlite intenta abrirlo en el cwd del
proceso de test, que puede estar en un filesystem sincronizado sin soporte de locking (esto
causaba "disk I/O error" en este sandbox). Ese engine de produccion solo se usa en el evento
de startup (init_db); las queries reales de cada test van por el engine de test (ver mas abajo).

Ademas, cada test recibe su propia base SQLite aislada sobreescribiendo la dependency
`get_db` (patron estandar de FastAPI), en vez de depender de mas variables de entorno.
"""
import os
import tempfile

os.environ.setdefault("DATABASE_URL", f"sqlite:///{tempfile.gettempdir()}/prescriptive_app_default.db")
os.environ.setdefault("DATASETS_DIR", tempfile.mkdtemp())
os.environ.setdefault("MODELS_DIR", tempfile.mkdtemp())

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base, get_db
from app.main import app


@pytest.fixture()
def client():
    db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False, dir=tempfile.gettempdir()).name
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def company_api_key(client) -> str:
    unique_email = f"demo-{uuid.uuid4().hex[:8]}@empresa.com"
    resp = client.post("/api/v1/companies", json={"name": "Empresa Demo", "contact_email": unique_email})
    assert resp.status_code == 201, f"email={unique_email} body={resp.text}"
    return resp.json()["api_key"]
