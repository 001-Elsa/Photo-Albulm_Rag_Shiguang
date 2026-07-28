from pathlib import Path

from fastapi.testclient import TestClient

from shiguang import personal_api as api
from shiguang.config import Config


def _app(monkeypatch, tmp_path: Path, **overrides):
    cfg = Config(
        embed_backend="demo",
        enable_ocr=False,
        enable_faces=False,
        **overrides,
    )
    monkeypatch.setattr(api.Config, "load", classmethod(lambda cls: cfg))
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        api,
        "get_paths",
        lambda: {
            "db": tmp_path / "shiguang.db",
            "thumbs": tmp_path / "thumbs",
            "models": tmp_path / "models",
        },
    )
    return api.create_app()


def test_personal_mode_does_not_create_auth_secret(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, auth_enabled=False)
    with TestClient(app) as client:
        assert client.get("/livez").status_code == 200
        assert client.get("/readyz").status_code == 200
        assert client.get("/metrics").status_code == 404
    assert not (tmp_path / "secret.key").exists()
    assert not (tmp_path / "admin_initial_password.txt").exists()


def test_registration_metrics_and_disabled_user_are_enforced(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SHIGUANG_BOOTSTRAP_ADMIN_PASSWORD", "correct-horse-battery")
    app = _app(
        monkeypatch,
        tmp_path,
        auth_enabled=True,
        allow_public_registration=False,
        expose_metrics=True,
    )
    with TestClient(app) as client:
        assert client.get("/api/auth/config").json() == {
            "auth_enabled": True,
            "public_registration": False,
        }
        assert client.post(
            "/api/register", json={"username": "new-user", "password": "password123"}
        ).status_code == 401
        assert client.get("/metrics").status_code == 401

        response = client.post(
            "/api/login",
            json={"username": "admin", "password": "correct-horse-battery"},
        )
        assert response.status_code == 200
        assert "SameSite=strict" in response.headers["set-cookie"]
        assert client.get("/metrics").status_code == 200

        app.state.db.execute(
            "UPDATE users SET disabled=1 WHERE username='admin'"
        )
        assert client.get("/api/me").status_code == 401
