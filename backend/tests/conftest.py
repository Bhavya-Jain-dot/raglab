import os
import tempfile
import uuid
from pathlib import Path

_db_path=Path(tempfile.gettempdir())/f"raglab-tests-{uuid.uuid4().hex}.db"
os.environ["DATABASE_URL"]=f"sqlite:///{_db_path.as_posix()}"
os.environ["SECRET_KEY"]="test-only-secret-that-is-long-enough"

import pytest
from fastapi.testclient import TestClient
from app.database import Base, engine
from app.main import app

@pytest.fixture(scope="session")
def client():
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as value:
        yield value
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    _db_path.unlink(missing_ok=True)
