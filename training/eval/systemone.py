"""Minimal client for the System One API (hosted Jev, or a compatible local server)."""

import time

import httpx


class SystemOne:
    def __init__(self, api_key: str, base_url: str = "https://api.typesafe.ai/v1",
                 model: str = "jev-latest"):
        self._model = model
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )

    def ask(self, state, questions: dict) -> dict:
        """Send one request; returns the `answers` map keyed by question id."""
        body = {"model": self._model, "state": state, "questions": questions}
        for attempt in range(4):
            resp = self._client.post("/systemone", json=body)
            if resp.status_code in (429, 529) and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()["answers"]
        raise RuntimeError("unreachable")
