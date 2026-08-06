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
    if not isinstance(steps, list) or not all(
        isinstance(step, Mapping) for step in steps
    ):
        raise ValueError("recipe steps must be a list of objects")
    return Recipe(steps=tuple(snapshot_step(step) for step in steps))


def snapshot_step(step: Mapping[str, object]) -> dict[str, object]:
    try:
        value = json.loads(json.dumps(dict(step), ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("recipe step must be JSON-compatible") from exc
    if not isinstance(value, dict):
        raise ValueError("recipe step must be an object")
    _reject_transient_browser_identity(value)
    _validate_browser_action_schema(value)
    return value


def _reject_transient_browser_identity(step: Mapping[str, object]) -> None:
    from browser_reuse.adapters.browser import transient_identity_keys

    forbidden_keys = transient_identity_keys()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            if forbidden_keys.intersection(value):
                raise ValueError("recipe step contains transient browser identity")
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(step)
    target = step.get("target")
    if not isinstance(target, Mapping):
        return
    if target.get("by") == "ref":
        raise ValueError("snapshot-local browser refs cannot enter a recipe")


def _validate_browser_action_schema(step: Mapping[str, object]) -> None:
    if step.get("op") not in {"click", "fill", "select_option"}:
        return
    from browser_reuse.adapters.browser import action_from_step

    try:
        action_from_step(step)
    except ValueError as exc:
        raise ValueError("recipe contains a noncanonical browser action") from exc
