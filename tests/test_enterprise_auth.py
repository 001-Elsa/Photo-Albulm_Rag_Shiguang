from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from shiguang.api.request_metadata import client_ip


def _client_ip_app() -> FastAPI:
    app = FastAPI()

    @app.get("/")
    def read_ip(request: Request) -> dict[str, str | None]:
        return {"ip": client_ip(request)}

    return app


def test_client_ip_keeps_only_valid_addresses() -> None:
    client = TestClient(_client_ip_app())

    assert client.get("/").json() == {"ip": None}
    assert client.get("/", headers={"X-Forwarded-For": "203.0.113.7, 10.0.0.1"}).json() == {
        "ip": "203.0.113.7"
    }
    assert client.get("/", headers={"X-Forwarded-For": "not-an-ip"}).json() == {
        "ip": None
    }
