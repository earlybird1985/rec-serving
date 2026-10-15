"""Build the backend named in the config."""
from __future__ import annotations

from pathlib import Path

from backends.beam import BeamBackend
from backends.toy import ToyBackend


def build_backend(cfg: dict):
    name = str(cfg.get("backend", "toy"))
    if name == "toy":
        return ToyBackend(n_items=int(cfg.get("n_items", 32)))
    if name == "beam":
        return BeamBackend(
            n_items=int(cfg.get("n_items", 32)),
            a3_checkpoint=cfg.get("a3_checkpoint"),
            a3_src=cfg.get("a3_src"),
        )
    if name == "hstu":
        from backends.hstu import HSTUBackend

        path = cfg.get("hstu_checkpoint")
        if not path or not Path(path).exists():
            raise FileNotFoundError(
                f"hstu_checkpoint missing ({path}). Train A6 or set backend: toy."
            )
        return HSTUBackend(path, catalog_chunk=int(cfg.get("catalog_chunk", 4096)))
    raise ValueError(f"unknown backend {name!r}")
