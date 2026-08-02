import time

import pytest
from fastapi.testclient import TestClient

# NOTE: api.server (and its torch dependency) is imported lazily inside the
# fixtures so torch-free unit tests (test_auth_unit.py) stay lightweight.


def _wait_for_ready(client, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get("/ready")
        if resp.status_code == 200 and resp.json().get("ready"):
            return
        time.sleep(0.1)
    raise RuntimeError("Server did not become ready")


@pytest.fixture(scope="module")
def app():
    """A mock-mode app with no auth keys — all /v1 endpoints are public."""
    from api.settings import Settings
    from api.server import create_app

    return create_app(
        Settings(
            mock_inference=True,
            auth_keys_file="/nonexistent/authorized_keys.json",
        )
    )


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app) as c:
        _wait_for_ready(c)
        yield c


# Re-export for test_server_mock.py to use a simpler import path
__all__ = ["app", "client"]
