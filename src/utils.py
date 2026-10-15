"""Config, seeding, JSON."""
from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)
