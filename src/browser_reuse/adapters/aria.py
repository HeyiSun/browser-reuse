"""Playwright AI-ARIA observation grounded to durable browser actions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Protocol

from browser_reuse.core import Observation

from .browser import (
    BrowserAction,
    Click,
    CssTarget,
    Fill,
    RoleTarget,
    action_from_step,
    action_to_step,
    execute_browser_action,
)


_NODE_PATTERN = re.compile(
    r'^\s*-\s+(?P<role>[a-z]+)(?:\s+"(?P<name>(?:[^"\\]|\\.)*)")?'
    r'.*?\[ref=(?P<ref>(?:f\d+)?e\d+)\]'
)
_SNAPSHOT_DEPTH = 12
_MAX_SNAPSHOT_OPTIONS = 20
_MAX_SELECT_OPTIONS = 40
_POLL_MS = 200
_MIN_POLLS = 5
_STABLE_MATCHES = 2
_MAX_POLLS = 40


class _Locator(Protocol):
    def aria_snapshot(self, *, mode: str, depth: int) -> str: ...

    def click(self) -> None: ...

    def count(self) -> int: ...

    def evaluate(self, expression: str): ...

    def fill(self, value: str) -> None: ...

    def get_attribute(self, name: str) -> str | None: ...

    def input_value(self) -> str: ...

    def is_editable(self) -> bool: ...

    def is_enabled(self) -> bool: ...

    def is_visible(self) -> bool: ...

    def select_option(self, *, label: str) -> None: ...


class _Page(Protocol):
    url: str

    def get_by_role(
        self,
        role: str,
        *,
        name: str,
        exact: bool,
    ) -> _Locator: ...

    def locator(self, selector: str) -> _Locator: ...

    def title(self) -> str: ...

    def wait_for_timeout(self, timeout: float) -> None: ...


class GenericBrowserAdapter:
    """Observe and act without site selectors, stages, or answer constants."""

    def __init__(self, page: _Page) -> None:
        self._page = page
        self._pending_requests: set[object] = set()
        on = getattr(page, "on", None)
        if callable(on):
            on("request", self._request_started)
            on("requestfinished", self._request_finished)
            on("requestfailed", self._request_finished)

    def observe(self) -> Observation:
        snapshot = self._snapshot()
        return Observation(
            data={
                "url": self._page.url,
                "title": self._page.title(),
                "snapshot": snapshot,
                "controls": self._ground_controls(snapshot),
            }
        )

    def execute(self, step: Mapping[str, object]) -> None:
        action = action_from_step(step)
        locator = self._resolve(action)
        if locator.count() != 1:
            raise ValueError("browser target is no longer unique")
        if not locator.is_visible() or not locator.is_enabled():
            raise ValueError("browser target is no longer actionable")
        if isinstance(action, Fill) and not locator.is_editable():
            raise ValueError("browser fill target is no longer editable")
        execute_browser_action(self._page, action)
        self._wait_until_stable()

    def _snapshot(self) -> str:
        return _compress_snapshot(
            self._page.locator("body").aria_snapshot(
                mode="ai",
                depth=_SNAPSHOT_DEPTH,
            )
        )

    def _ground_controls(self, snapshot: str) -> dict[str, dict[str, object]]:
        controls: dict[str, dict[str, object]] = {}
        for line in snapshot.splitlines():
            match = _NODE_PATTERN.match(line)
            if match is None:
                continue
            role = match.group("role")
            name = _decode_name(match.group("name"))
            ref = match.group("ref")
            locator = self._page.locator(f"aria-ref={ref}")
            if (
                locator.count() != 1
                or not locator.is_visible()
                or not locator.is_enabled()
            ):
                continue
            tag = str(locator.evaluate("element => element.tagName.toLowerCase()"))
            operation = _operation(role, tag, locator)
            if operation is None:
                continue
            target = self._durable_target(locator, role, name)
            if target is None:
                continue
            control: dict[str, object] = {
                "op": operation,
                "name": name,
                "target": action_to_step(Click(target))["target"],
            }
            if operation == "fill":
                control["value"] = locator.input_value()
            elif operation == "select_option":
                select_state = locator.evaluate(
                    """element => ({
                        value: element.selectedOptions[0]?.textContent?.trim() || '',
                        labels: Array.from(element.options)
                            .filter(option =>
                                !option.disabled && !option.hidden && option.value
                            )
                            .map(option => option.textContent.trim())
                    })"""
                )
                if not isinstance(select_state, Mapping):
                    continue
                labels = tuple(str(label) for label in select_state.get("labels", ()))
                if (
                    len(labels) > _MAX_SELECT_OPTIONS
                    or len(labels) != len(set(labels))
                ):
                    continue
                control["value"] = str(select_state.get("value", ""))
                control["labels"] = labels
            controls[ref] = control
        return controls

    def _durable_target(
        self,
        locator: _Locator,
        role: str,
        name: str,
    ) -> CssTarget | RoleTarget | None:
        if name:
            by_role = self._page.get_by_role(role, name=name, exact=True)
            if by_role.count() == 1:
                return RoleTarget(role, name)
        for attribute in ("id", "data-testid", "name"):
            value = locator.get_attribute(attribute)
            if not value:
                continue
            selector = f"[{attribute}={json.dumps(value)}]"
            if self._page.locator(selector).count() == 1:
                return CssTarget(selector)
        return None

    def _resolve(self, action: BrowserAction) -> _Locator:
        target = action.target
        if isinstance(target, CssTarget):
            return self._page.locator(target.selector)
        return self._page.get_by_role(
            target.role,
            name=target.name,
            exact=True,
        )

    def _wait_until_stable(self) -> None:
        previous: tuple[str, str] | None = None
        stable_matches = 0
        for poll in range(_MAX_POLLS):
            self._page.wait_for_timeout(_POLL_MS)
            current = (self._page.url, self._snapshot())
            if not self._pending_requests and current == previous:
                stable_matches += 1
            else:
                previous = current
                stable_matches = 0
            if poll + 1 >= _MIN_POLLS and stable_matches >= _STABLE_MATCHES:
                return
        raise TimeoutError("page did not reach a stable public state")

    def _request_started(self, request: object) -> None:
        self._pending_requests.add(request)

    def _request_finished(self, request: object) -> None:
        self._pending_requests.discard(request)


def _decode_name(value: str | None) -> str:
    if value is None:
        return ""
    return str(json.loads(f'"{value}"'))


def _compress_snapshot(snapshot: str) -> str:
    lines = snapshot.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        output.append(line)
        if "- combobox" not in line:
            index += 1
            continue
        indent = len(line) - len(line.lstrip())
        end = index + 1
        while end < len(lines):
            nested_indent = len(lines[end]) - len(lines[end].lstrip())
            if nested_indent <= indent:
                break
            end += 1
        nested = lines[index + 1 : end]
        option_lines = [line for line in nested if line.lstrip().startswith("- option")]
        if len(option_lines) <= _MAX_SNAPSHOT_OPTIONS:
            output.extend(nested)
        else:
            kept = option_lines[:_MAX_SNAPSHOT_OPTIONS]
            selected = [line for line in option_lines if "[selected]" in line]
            for selected_line in selected:
                if selected_line not in kept:
                    kept.append(selected_line)
            output.extend(kept)
            omitted = len(option_lines) - len(kept)
            output.append(
                " " * (indent + 2)
                + f"- text: [{omitted} additional options omitted]"
            )
        index = end
    return "\n".join(output)


def _operation(role: str, tag: str, locator: _Locator) -> str | None:
    if role == "combobox" and tag == "select":
        return "select_option"
    if role in {"searchbox", "spinbutton", "textbox"} and locator.is_editable():
        return "fill"
    if role in {
        "button",
        "checkbox",
        "link",
        "menuitem",
        "option",
        "radio",
        "switch",
        "tab",
    }:
        return "click"
    return None
