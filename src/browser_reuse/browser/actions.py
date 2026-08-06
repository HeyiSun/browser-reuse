"""Typed, site-neutral browser actions and their strict step codec."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from .targets import (
    BrowserTarget,
    DurableTarget,
    _target_from_mapping,
    _target_to_mapping,
)


@dataclass(frozen=True)
class Click:
    """Click the unique node confirmed by a durable target."""

    target: BrowserTarget

    def __post_init__(self) -> None:
        _validate_target(self.target)


@dataclass(frozen=True)
class Fill:
    """Replace an editable target's text with the given value."""

    target: BrowserTarget
    value: str

    def __post_init__(self) -> None:
        _validate_target(self.target)
        if not isinstance(self.value, str):
            raise ValueError("fill value must be a string")


@dataclass(frozen=True)
class SelectOption:
    """Choose one visible label from a native select control."""

    target: BrowserTarget
    label: str

    def __post_init__(self) -> None:
        _validate_target(self.target)
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("select option label must not be empty")


BrowserAction: TypeAlias = Click | Fill | SelectOption


def action_to_step(action: BrowserAction) -> dict[str, object]:
    """Encode a typed action as the canonical recipe step mapping."""

    target = _target_to_mapping(action.target)
    if isinstance(action, Click):
        return {"op": "click", "target": target}
    if isinstance(action, Fill):
        return {"op": "fill", "target": target, "value": action.value}
    return {"op": "select_option", "target": target, "label": action.label}


def action_from_step(step: Mapping[str, object]) -> BrowserAction:
    """Decode a canonical step and reject unknown or missing fields."""

    operation = step.get("op")
    expected_keys = {
        "click": {"op", "target"},
        "fill": {"op", "target", "value"},
        "select_option": {"op", "target", "label"},
    }
    if operation not in expected_keys or set(step) != expected_keys[operation]:
        raise ValueError(f"invalid browser action fields: {operation!r}")
    target = _target_from_mapping(step.get("target"))
    if operation == "click":
        return Click(target)
    if operation == "fill":
        value = step.get("value")
        if not isinstance(value, str):
            raise ValueError("fill requires a string value")
        return Fill(target, value)
    label = step.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError("select_option requires a non-empty label")
    return SelectOption(target, label)


def _validate_target(target: object) -> None:
    if not isinstance(target, DurableTarget):
        raise ValueError("browser action target has an unsupported type")
