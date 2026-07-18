"""Typed, site-neutral browser actions and their Playwright execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeAlias


@dataclass(frozen=True)
class CssTarget:
    selector: str


@dataclass(frozen=True)
class RoleTarget:
    role: str
    name: str


BrowserTarget: TypeAlias = CssTarget | RoleTarget


@dataclass(frozen=True)
class Click:
    target: BrowserTarget


@dataclass(frozen=True)
class Fill:
    target: BrowserTarget
    value: str


@dataclass(frozen=True)
class SelectOption:
    target: BrowserTarget
    label: str


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


def _target_to_mapping(target: BrowserTarget) -> dict[str, str]:
    if isinstance(target, CssTarget):
        return {"by": "css", "selector": target.selector}
    return {"by": "role", "role": target.role, "name": target.name}


def _target_from_mapping(value: object) -> BrowserTarget:
    if not isinstance(value, Mapping):
        raise ValueError("browser action requires a target mapping")
    strategy = value.get("by")
    if strategy == "css":
        selector = value.get("selector")
        if not isinstance(selector, str) or not selector:
            raise ValueError("css target requires a non-empty selector")
        return CssTarget(selector)
    if strategy in {None, "role"}:
        role = value.get("role")
        name = value.get("name")
        if not isinstance(role, str) or not role:
            raise ValueError("role target requires a non-empty role")
        if not isinstance(name, str) or not name:
            raise ValueError("role target requires a non-empty name")
        return RoleTarget(role, name)
    raise ValueError(f"unknown target strategy: {strategy!r}")
