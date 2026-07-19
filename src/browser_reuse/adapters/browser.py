"""Typed, site-neutral browser actions and their Playwright execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeAlias


@dataclass(frozen=True)
class CssTarget:
    selector: str

    def __post_init__(self) -> None:
        if not self.selector:
            raise ValueError("css selector must not be empty")


@dataclass(frozen=True)
class RoleTarget:
    role: str
    name: str

    def __post_init__(self) -> None:
        if not self.role or not self.name:
            raise ValueError("role and name must not be empty")


BrowserTarget: TypeAlias = CssTarget | RoleTarget


@dataclass(frozen=True)
class Click:
    target: BrowserTarget

    def __post_init__(self) -> None:
        _validate_target(self.target)


@dataclass(frozen=True)
class Fill:
    target: BrowserTarget
    value: str

    def __post_init__(self) -> None:
        _validate_target(self.target)
        if not isinstance(self.value, str):
            raise ValueError("fill value must be a string")


@dataclass(frozen=True)
class SelectOption:
    target: BrowserTarget
    label: str

    def __post_init__(self) -> None:
        _validate_target(self.target)
        if not self.label:
            raise ValueError("select option label must not be empty")


BrowserAction: TypeAlias = Click | Fill | SelectOption


class _Locator(Protocol):
    def click(self) -> None: ...

    def fill(self, value: str) -> None: ...

    def select_option(self, *, label: str) -> None: ...


class _Page(Protocol):
    def locator(self, selector: str) -> _Locator: ...

    def get_by_role(
        self,
        role: str,
        *,
        name: str,
        exact: bool,
    ) -> _Locator: ...


def execute_browser_action(page: _Page, action: BrowserAction) -> None:
    locator = _resolve(page, action.target)
    if isinstance(action, Click):
        locator.click()
    elif isinstance(action, Fill):
        locator.fill(action.value)
    else:
        locator.select_option(label=action.label)


def action_to_step(action: BrowserAction) -> dict[str, object]:
    target = _target_to_mapping(action.target)
    if isinstance(action, Click):
        return {"op": "click", "target": target}
    if isinstance(action, Fill):
        return {"op": "fill", "target": target, "value": action.value}
    return {"op": "select_option", "target": target, "label": action.label}


def action_from_step(step: Mapping[str, object]) -> BrowserAction:
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
    if operation == "select_option":
        label = step.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError("select_option requires a non-empty label")
        return SelectOption(target, label)
    raise ValueError(f"unknown browser action: {operation!r}")


def execute_browser_step(page: _Page, step: Mapping[str, object]) -> None:
    execute_browser_action(page, action_from_step(step))


def _resolve(page: _Page, target: BrowserTarget) -> _Locator:
    if isinstance(target, CssTarget):
        return page.locator(target.selector)
    return page.get_by_role(target.role, name=target.name, exact=True)


def _validate_target(target: object) -> None:
    if not isinstance(target, (CssTarget, RoleTarget)):
        raise ValueError("browser action target has an unsupported type")


def _target_to_mapping(target: BrowserTarget) -> dict[str, str]:
    if isinstance(target, CssTarget):
        return {"by": "css", "selector": target.selector}
    return {"by": "role", "role": target.role, "name": target.name}


def _target_from_mapping(value: object) -> BrowserTarget:
    if not isinstance(value, Mapping):
        raise ValueError("browser action requires a target mapping")
    strategy = value.get("by")
    if strategy == "css":
        if set(value) != {"by", "selector"}:
            raise ValueError("invalid css target fields")
        selector = value.get("selector")
        if not isinstance(selector, str) or not selector:
            raise ValueError("css target requires a non-empty selector")
        return CssTarget(selector)
    if strategy in {None, "role"}:
        expected = {"role", "name"} if strategy is None else {"by", "role", "name"}
        if set(value) != expected:
            raise ValueError("invalid role target fields")
        role = value.get("role")
        name = value.get("name")
        if not isinstance(role, str) or not role:
            raise ValueError("role target requires a non-empty role")
        if not isinstance(name, str) or not name:
            raise ValueError("role target requires a non-empty name")
        return RoleTarget(role, name)
    raise ValueError(f"unknown target strategy: {strategy!r}")
