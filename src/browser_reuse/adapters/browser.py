"""Typed, site-neutral browser actions and their Playwright execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Protocol, TypeAlias


_STABLE_WITNESS_ATTRIBUTES = frozenset(
    {
        "aria-label",
        "data-cy",
        "data-qa",
        "data-test",
        "data-test-id",
        "data-testid",
        "id",
        "name",
        "placeholder",
    }
)
_TAG_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
_XPATH_PREFIX_PATTERN = re.compile(r"^(?:xpath\s*=|//|/|\.//|\(\s*//)", re.IGNORECASE)
_POSITIONAL_SELECTOR_PATTERN = re.compile(
    r":(?:nth(?:-child|-last-child|-of-type|-last-of-type)?\s*\(|"
    r"first-child|last-child|only-child|first-of-type|last-of-type|only-of-type)",
    re.IGNORECASE,
)
_FIRST_CLASS_PATTERN = re.compile(r"\.first(?![a-z0-9_-])", re.IGNORECASE)


@dataclass(frozen=True)
class ElementWitness:
    """Stable public evidence used to reject a changed CSS target on replay."""

    tag: str
    role: str | None = None
    name: str | None = None
    attributes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tag, str) or not _TAG_PATTERN.fullmatch(self.tag):
            raise ValueError("witness tag must be a lowercase HTML tag")
        _validate_optional_witness_text("role", self.role)
        _validate_optional_witness_text("name", self.name)

        attributes: list[tuple[str, str]] = []
        seen: set[str] = set()
        if not isinstance(self.attributes, tuple):
            raise ValueError("witness attributes must be a tuple of string pairs")
        for item in self.attributes:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not isinstance(item[1], str)
                or not item[1]
            ):
                raise ValueError("witness attributes must contain non-empty string pairs")
            key, value = item
            if key not in _STABLE_WITNESS_ATTRIBUTES:
                raise ValueError(f"unsupported witness attribute: {key!r}")
            if key in seen:
                raise ValueError(f"duplicate witness attribute: {key!r}")
            seen.add(key)
            attributes.append((key, value))
        object.__setattr__(self, "attributes", tuple(sorted(attributes)))


@dataclass(frozen=True)
class CssTarget:
    selector: str
    witness: ElementWitness

    def __post_init__(self) -> None:
        _validate_durable_css_selector(self.selector)
        if not isinstance(self.witness, ElementWitness):
            raise ValueError("css target requires an element witness")


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


def _target_to_mapping(target: BrowserTarget) -> dict[str, object]:
    if isinstance(target, CssTarget):
        return {
            "by": "css",
            "selector": target.selector,
            "witness": _witness_to_mapping(target.witness),
        }
    return {"by": "role", "role": target.role, "name": target.name}


def _target_from_mapping(value: object) -> BrowserTarget:
    if not isinstance(value, Mapping):
        raise ValueError("browser action requires a target mapping")
    strategy = value.get("by")
    if strategy == "css":
        if set(value) != {"by", "selector", "witness"}:
            raise ValueError("invalid css target fields")
        selector = value.get("selector")
        if not isinstance(selector, str) or not selector:
            raise ValueError("css target requires a non-empty selector")
        return CssTarget(selector, _witness_from_mapping(value.get("witness")))
    if strategy == "role":
        if set(value) != {"by", "role", "name"}:
            raise ValueError("invalid role target fields")
        role = value.get("role")
        name = value.get("name")
        if not isinstance(role, str) or not role:
            raise ValueError("role target requires a non-empty role")
        if not isinstance(name, str) or not name:
            raise ValueError("role target requires a non-empty name")
        return RoleTarget(role, name)
    raise ValueError(f"unknown target strategy: {strategy!r}")


def _witness_to_mapping(witness: ElementWitness) -> dict[str, object]:
    return {
        "tag": witness.tag,
        "role": witness.role,
        "name": witness.name,
        "attributes": dict(witness.attributes),
    }


def _witness_from_mapping(value: object) -> ElementWitness:
    if not isinstance(value, Mapping):
        raise ValueError("css target requires a witness mapping")
    if set(value) != {"tag", "role", "name", "attributes"}:
        raise ValueError("invalid witness fields")

    tag = value.get("tag")
    role = value.get("role")
    name = value.get("name")
    attributes = value.get("attributes")
    if not isinstance(tag, str):
        raise ValueError("witness requires a tag")
    if role is not None and not isinstance(role, str):
        raise ValueError("witness role must be a string or null")
    if name is not None and not isinstance(name, str):
        raise ValueError("witness name must be a string or null")
    if not isinstance(attributes, Mapping):
        raise ValueError("witness attributes must be a mapping")
    pairs: list[tuple[str, str]] = []
    for key, attribute_value in attributes.items():
        if not isinstance(key, str) or not isinstance(attribute_value, str):
            raise ValueError("witness attributes must map strings to strings")
        pairs.append((key, attribute_value))
    return ElementWitness(tag, role, name, tuple(pairs))


def _validate_optional_witness_text(field: str, value: object) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"witness {field} must be a non-empty string or null")


def _validate_durable_css_selector(selector: object) -> None:
    if not isinstance(selector, str) or not selector.strip():
        raise ValueError("css selector must not be empty")
    normalized = selector.strip()
    if normalized != selector:
        raise ValueError("css selector must not contain outer whitespace")
    if _XPATH_PREFIX_PATTERN.search(normalized):
        raise ValueError("XPath is not a durable CSS selector")
    if ">>" in normalized:
        raise ValueError("chained Playwright selectors are not durable")
    if "," in normalized:
        raise ValueError("CSS selector lists are not durable")
    if _POSITIONAL_SELECTOR_PATTERN.search(normalized) or _FIRST_CLASS_PATTERN.search(
        normalized
    ):
        raise ValueError("positional CSS selectors are not durable")
