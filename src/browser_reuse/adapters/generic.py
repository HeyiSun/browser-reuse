"""Site-neutral browser adapter over DOM+AX snapshot-local refs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from browser_reuse.core import Observation
from browser_reuse.interfaces import ActionDispatchedError

from .browser import (
    BrowserAction,
    Click,
    DurableTarget,
    Fill,
    SelectOption,
    action_from_step,
    action_to_step,
)
from .grounding import DOM_REVISION_SCRIPT, DomAxGrounder


_POLL_MS = 200
_MIN_POLLS = 3
_STABLE_MATCHES = 2
_MAX_POLLS = 40


class _Locator(Protocol):
    def click(self) -> None: ...

    def count(self) -> int: ...

    def evaluate(self, expression: str, arg: object | None = None): ...

    def fill(self, value: str) -> None: ...

    def get_attribute(self, name: str) -> str | None: ...

    def is_editable(self) -> bool: ...

    def is_enabled(self) -> bool: ...

    def is_visible(self) -> bool: ...

    def select_option(self, *, label: str) -> None: ...


class _Page(Protocol):
    url: str

    def evaluate(self, expression: str): ...

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
    """Observe through DOM+AX, execute live refs, and replay witnessed targets."""

    def __init__(
        self,
        page: _Page,
        *,
        grounder: DomAxGrounder | None = None,
    ) -> None:
        self._page = page
        self._grounder = grounder or DomAxGrounder(page)  # type: ignore[arg-type]
        self._pending_requests: set[object] = set()
        on = getattr(page, "on", None)
        if callable(on):
            on("request", self._request_started)
            on("requestfinished", self._request_finished)
            on("requestfailed", self._request_finished)

    def observe(self) -> Observation:
        snapshot = self._grounder.capture()
        return Observation(
            data={
                "url": self._page.url,
                "title": self._page.title(),
                "snapshot": snapshot.text,
                "snapshot_token": snapshot.token,
                "controls": snapshot.controls,
                "diagnostics": snapshot.diagnostics,
            }
        )

    def execute(self, step: Mapping[str, object]) -> Mapping[str, object] | None:
        target = step.get("target")
        effective_step: Mapping[str, object] | None = None
        if isinstance(target, Mapping) and target.get("by") == "ref":
            self._execute_ref_step(step, target)
        else:
            effective_step = self._execute_durable_step(step)
        try:
            self._wait_until_stable()
        except Exception as exc:
            raise ActionDispatchedError(
                "browser action was dispatched but the page did not settle"
            ) from exc
        return effective_step

    def _execute_ref_step(
        self,
        step: Mapping[str, object],
        target: Mapping[str, object],
    ) -> None:
        operation = step.get("op")
        expected = {
            "click": {"op", "target"},
            "fill": {"op", "target", "value"},
            "select_option": {"op", "target", "label"},
        }
        if operation not in expected or set(step) != expected[operation]:
            raise ValueError("invalid snapshot-ref browser action")
        if set(target) != {"by", "snapshot", "ref"}:
            raise ValueError("invalid snapshot-ref target")
        token = target.get("snapshot")
        ref = target.get("ref")
        if not isinstance(token, str) or not isinstance(ref, str):
            raise ValueError("snapshot-ref target requires string identity")
        label = step.get("label") if operation == "select_option" else None
        locator = self._grounder.resolve_ref(
            token,
            ref,
            str(operation),
            label=label if isinstance(label, str) else None,
        )
        _execute_locator(locator, operation, step)

    def _execute_durable_step(
        self,
        step: Mapping[str, object],
    ) -> Mapping[str, object]:
        action = action_from_step(step)
        label = action.label if isinstance(action, SelectOption) else None
        locator, effective_target = self._grounder.resolve_durable_target(
            action.target,
            step.get("op", ""),
            label=label,
        )
        _execute_locator(locator, step.get("op"), step)
        return action_to_step(_action_with_target(action, effective_target))

    def _wait_until_stable(self) -> None:
        previous: tuple[str, str, int] | None = None
        stable_matches = 0
        for poll in range(_MAX_POLLS):
            self._page.wait_for_timeout(_POLL_MS)
            state = self._page.evaluate(DOM_REVISION_SCRIPT)
            if not isinstance(state, Mapping):
                raise RuntimeError("browser did not return DOM quiet state")
            ready = state.get("ready")
            version = state.get("version")
            if not isinstance(ready, str) or not isinstance(version, int):
                raise RuntimeError("browser returned invalid DOM quiet state")
            current = (self._page.url, ready, version)
            if (
                not self._pending_requests
                and ready != "loading"
                and current == previous
            ):
                stable_matches += 1
            else:
                previous = current
                stable_matches = 0
            if poll + 1 >= _MIN_POLLS and stable_matches >= _STABLE_MATCHES:
                return
        raise TimeoutError("page did not reach a bounded network and DOM quiet state")

    def _request_started(self, request: object) -> None:
        if getattr(request, "resource_type", "") in {"websocket", "eventsource"}:
            return
        self._pending_requests.add(request)

    def _request_finished(self, request: object) -> None:
        self._pending_requests.discard(request)


def _execute_locator(
    locator: _Locator,
    operation: object,
    step: Mapping[str, object],
) -> None:
    if operation == "click":
        locator.click()
        return
    if operation == "fill":
        value = step.get("value")
        if not isinstance(value, str):
            raise ValueError("fill requires a string value")
        locator.fill(value)
        return
    if operation == "select_option":
        label = step.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError("select_option requires a non-empty label")
        locator.select_option(label=label)
        return
    raise ValueError(f"unknown browser action: {operation!r}")


def _action_with_target(
    action: BrowserAction,
    target: DurableTarget,
) -> BrowserAction:
    if isinstance(action, Click):
        return Click(target)
    if isinstance(action, Fill):
        return Fill(target, action.value)
    return SelectOption(target, action.label)
