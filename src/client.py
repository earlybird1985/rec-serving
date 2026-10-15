"""One-request helper for the bench and the notebook."""
from __future__ import annotations

import json
from urllib.request import Request, urlopen


def post(base_url: str, path: str, body: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get(base_url: str, path: str, timeout: float = 10.0) -> dict:
    with urlopen(base_url.rstrip("/") + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
