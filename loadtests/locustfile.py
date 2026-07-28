"""企业 API 负载场景。

启动前设置：
  SHIGUANG_LOAD_USERNAME / SHIGUANG_LOAD_PASSWORD / SHIGUANG_LOAD_ORG_ID

示例：
  locust -f loadtests/locustfile.py --host http://127.0.0.1:8626
  locust -f loadtests/locustfile.py --headless -u 50 -r 5 -t 5m \
    --host http://127.0.0.1:8626 --csv reports/locust-50
"""
from __future__ import annotations

import os
import random

from locust import HttpUser, between, task

QUERIES = (
    "海边日落",
    "2024 年春天的樱花",
    "高铁票截图",
    "会议白板文字",
    "和同事的合影",
)


class SearchUser(HttpUser):
    wait_time = between(0.2, 1.2)

    def on_start(self) -> None:
        username = os.environ["SHIGUANG_LOAD_USERNAME"]
        password = os.environ["SHIGUANG_LOAD_PASSWORD"]
        response = self.client.post(
            "/api/v1/auth/login",
            json={"username": username, "password": password},
            name="/api/v1/auth/login",
        )
        response.raise_for_status()
        self.token = response.json()["access_token"]
        self.organization_id = os.environ["SHIGUANG_LOAD_ORG_ID"]

    @task(9)
    def search(self) -> None:
        self.client.get(
            f"/api/v1/organizations/{self.organization_id}/search",
            params={"q": random.choice(QUERIES), "limit": 20},
            headers={"Authorization": f"Bearer {self.token}"},
            name="/api/v1/organizations/:id/search",
        )

    @task(1)
    def list_jobs(self) -> None:
        self.client.get(
            f"/api/v1/organizations/{self.organization_id}/jobs",
            headers={"Authorization": f"Bearer {self.token}"},
            name="/api/v1/organizations/:id/jobs",
        )
