import os
import time
import importlib
import pytest
from fastapi.testclient import TestClient


os.environ["GLM_TTS_MOCK_INFERENCE"] = "1"


def _load_server():
    os.environ["GLM_TTS_MOCK_INFERENCE"] = "1"
    os.environ["GLM_TTS_AUTH_KEYS_FILE"] = "/nonexistent/authorized_keys.json"
    import api.server as server
    importlib.reload(server)
    return server


def _wait_for_ready(client, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get("/ready")
        if resp.status_code == 200 and resp.json().get("ready"):
            return
        time.sleep(0.1)
    raise RuntimeError("Server did not become ready")


@pytest.fixture(scope="module")
def client():
    server = _load_server()
    with TestClient(server.app) as c:
        _wait_for_ready(c)
        yield c


# Re-export for test_server_mock.py to use a simpler import path
__all__ = ["client"]
