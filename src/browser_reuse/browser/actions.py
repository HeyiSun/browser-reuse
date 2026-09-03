"""Typed, site-neutral browser actions and their strict step codec."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias
from urllib.parse import urlsplit, urlunsplit

from .targets import (
    BrowserTarget,
    DurableTarget,
    _target_from_mapping,
    _target_to_mapping,
)


@dataclass(frozen=True)
class Appears:
    """Confirm a click when one exact named AX fact newly appears."""

    role: str
    name: str

    def __post_init__(self) -> None:
        if self.role not in {
            "alert",
            "button",
            "dialog",
            "heading",
            "link",
            "menuitem",
            "option",
            "status",
            "tab",
        }:
            raise ValueError("appears readback has an unsupported role")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("appears readback requires a name")


@dataclass(frozen=True)
class UrlIs:
    """Confirm a same-origin link at one exact relative path and query."""

    relative_url: str

    def __post_init__(self) -> None:
        if not isinstance(self.relative_url, str):
            raise ValueError("url readback requires a relative URL string")
        parsed = urlsplit(self.relative_url)
        canonical = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        if (
            parsed.scheme
            or parsed.netloc
            or parsed.fragment
            or not parsed.path.startswith("/")
            or canonical != self.relative_url
        ):
            raise ValueError(
                "url readback requires one canonical absolute path and query"
            )


@dataclass(frozen=True)
class Click:
    """Click the unique node confirmed by a durable target."""

    target: BrowserTarget
    readback: Appears | UrlIs | None = None

    def __post_init__(self) -> None:
        _validate_target(self.target)
        if self.readback is not None and not isinstance(
            self.readback, (Appears, UrlIs)
        ):
            raise ValueError("click readback has an unsupported condition")


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


@dataclass(frozen=True)
class ChooseComboboxOption:
    """Choose one exact option from a non-native combobox field."""

    target: BrowserTarget
    label: str

    def __post_init__(self) -> None:
        _validate_target(self.target)
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("combobox option label must not be empty")


@dataclass(frozen=True)
class SetChecked:
    """Make one checkbox or radio match a witnessed boolean state."""

    target: BrowserTarget
    checked: bool

    def __post_init__(self) -> None:
        _validate_target(self.target)
        if not isinstance(self.checked, bool):
            raise ValueError("set checked requires a boolean state")


BrowserAction: TypeAlias = (
    Click | Fill | SelectOption | ChooseComboboxOption | SetChecked
)


def action_to_step(action: BrowserAction) -> dict[str, object]:
    """Encode a typed action as the canonical recipe step mapping."""

    target = _target_to_mapping(action.target)
    if isinstance(action, Click):
        step: dict[str, object] = {"op": "click", "target": target}
        if isinstance(action.readback, Appears):
            step["readback"] = {
                "kind": "appears",
                "role": action.readback.role,
                "name": action.readback.name,
            }
        elif isinstance(action.readback, UrlIs):
            step["readback"] = {
                "kind": "url_is",
                "relative_url": action.readback.relative_url,
            }
        return step
    if isinstance(action, Fill):
        return {"op": "fill", "target": target, "value": action.value}
    if isinstance(action, SelectOption):
        return {"op": "select_option", "target": target, "label": action.label}
    if isinstance(action, ChooseComboboxOption):
        return {
            "op": "choose_combobox_option",
            "target": target,
            "label": action.label,
        }
    return {"op": "set_checked", "target": target, "checked": action.checked}


def action_from_step(step: Mapping[str, object]) -> BrowserAction:
    """Decode a canonical step and reject unknown or missing fields."""

    operation = step.get("op")
    expected_keys = {
        "fill": {"op", "target", "value"},
        "select_option": {"op", "target", "label"},
        "choose_combobox_option": {"op", "target", "label"},
        "set_checked": {"op", "target", "checked"},
    }
    if operation == "click":
        if set(step) not in (
            {"op", "target"},
            {"op", "target", "readback"},
        ):
            raise ValueError("invalid browser action fields: 'click'")
    elif operation not in expected_keys or set(step) != expected_keys[operation]:
        raise ValueError(f"invalid browser action fields: {operation!r}")
    target = _target_from_mapping(step.get("target"))
    if operation == "click":
        readback = step.get("readback")
        if readback is None:
            return Click(target)
        if not isinstance(readback, Mapping):
            raise ValueError("invalid click readback fields")
        if readback.get("kind") == "appears":
            if set(readback) != {"kind", "role", "name"}:
                raise ValueError("invalid click readback fields")
            role = readback.get("role")
            name = readback.get("name")
            if not isinstance(role, str) or not isinstance(name, str):
                raise ValueError("click readback role and name must be strings")
            return Click(target, Appears(role=role, name=name))
        if readback.get("kind") == "url_is":
            if set(readback) != {"kind", "relative_url"}:
                raise ValueError("invalid click readback fields")
            relative_url = readback.get("relative_url")
            if not isinstance(relative_url, str):
                raise ValueError("click url readback must be a string")
            return Click(target, UrlIs(relative_url))
        raise ValueError("unsupported click readback kind")
    if operation == "fill":
        value = step.get("value")
        if not isinstance(value, str):
            raise ValueError("fill requires a string value")
        return Fill(target, value)
    if operation == "set_checked":
        checked = step.get("checked")
        if not isinstance(checked, bool):
            raise ValueError("set_checked requires a boolean state")
        return SetChecked(target, checked)
    label = step.get("label")
    if not isinstance(label, str) or not label:
        raise ValueError(f"{operation} requires a non-empty label")
    if operation == "select_option":
        return SelectOption(target, label)
    return ChooseComboboxOption(target, label)


def _validate_target(target: object) -> None:
    if not isinstance(target, DurableTarget):
        raise ValueError("browser action target has an unsupported type")
