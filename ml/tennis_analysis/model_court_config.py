"""Single source of truth for model-court geometry (loaded from model_court.json)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).with_name("model_court.json")


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


def court() -> dict[str, Any]:
    return load_config()["court"]


def public_court_model() -> dict[str, Any]:
    """Subset for API + frontend (court drawing + singles bounds)."""
    c = court()
    return {
        "version": load_config().get("version", 1),
        "width": c["width"],
        "height": c["height"],
        "netY": c["netY"],
        "runoff": c["runoff"],
        "svgScale": c["svgScale"],
        "singles": c["singles"],
        "lines": c["lines"],
    }


def bounce_singles_call(x, y) -> str:
    s = court()["singles"]
    try:
        xf, yf = float(x), float(y)
    except (TypeError, ValueError):
        return "unknown"
    if s["xMin"] <= xf <= s["xMax"] and s["yMin"] <= yf <= s["yMax"]:
        return "in"
    return "out"
