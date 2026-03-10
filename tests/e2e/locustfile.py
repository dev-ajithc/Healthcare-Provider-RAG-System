"""Locust load test for the Healthcare Provider RAG API."""

import random
import uuid

from locust import HttpUser, between, task

SAMPLE_QUERIES = [
    "cardiologists in California accepting Medicare",
    "pediatricians in Texas accepting new patients",
    "psychiatrists in New York",
    "dermatologists accepting Medicaid in Florida",
    "family medicine doctors in Illinois",
    "neurologists in Washington state",
    "orthopedic surgeons in Ohio",
    "internal medicine physicians in Pennsylvania",
    "endocrinologists in Michigan accepting Aetna",
    "OB/GYN doctors accepting Medicare in Georgia",
    "doctors accepting new patients near me",
    "top rated specialists in California",
    "pulmonologists accepting Cigna in Texas",
    "rheumatologists in Massachusetts",
    "gastroenterologists accepting Humana in Florida",
]


class ProviderQueryUser(HttpUser):
    wait_time = between(1, 3)
    session_id: str = ""

    def on_start(self) -> None:
        self.session_id = str(uuid.uuid4())

    @task(10)
    def query(self) -> None:
        query_text = random.choice(SAMPLE_QUERIES)
        payload = {
            "query": query_text,
            "session_id": self.session_id,
            "hyde_enabled": False,
        }
        with self.client.post(
            "/query",
            json=payload,
            catch_response=True,
            name="/query",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "answer" not in data:
                    response.failure(
                        "Missing 'answer' in response"
                    )
                else:
                    response.success()
            elif response.status_code == 429:
                response.success()
            else:
                response.failure(
                    f"Unexpected status: {response.status_code}"
                )

    @task(2)
    def health_check(self) -> None:
        self.client.get("/health", name="/health")

    @task(1)
    def get_session(self) -> None:
        self.client.get(
            f"/session/{self.session_id}",
            name="/session/{id}",
        )
