"""Canonical JSON-compatible Recipe payload conversion."""

from __future__ import annotations

import json
from collections.abc import Mapping

from browser_reuse.core import Recipe


def recipe_to_payload(recipe: Recipe) -> dict[str, object]:
    return {"steps": [snapshot_step(step) for step in recipe.steps]}


def recipe_from_payload(payload: object) -> Recipe:
    if not isinstance(payload, Mapping) or set(payload) != {"steps"}:
        raise ValueError("recipe payload must contain exactly steps")
    steps = payload["steps"]
    if not isinstance(steps, list) or not all(isinstance(step, Mapping) for step in steps):
        raise ValueError("recipe steps must be a list of objects")
    return Recipe(steps=tuple(snapshot_step(step) for step in steps))


def snapshot_step(step: Mapping[str, object]) -> dict[str, object]:
    try:
        value = json.loads(json.dumps(dict(step), ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("recipe step must be JSON-compatible") from exc
    if not isinstance(value, dict):
        raise ValueError("recipe step must be an object")
    return value
