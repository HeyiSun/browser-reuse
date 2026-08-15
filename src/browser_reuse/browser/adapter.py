"""Site-neutral browser adapter over DOM+AX snapshot-local refs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from browser_reuse.core import Observation
from browser_reuse.interfaces import ActionDispatchedError, ActionNotCommittedError

from .actions import (
    BrowserAction,
    ChooseComboboxOption,
    Click,
    Fill,
    SelectOption,
    action_from_step,
    action_to_step,
)
from .dom_ax import (
    DOM_REVISION_SCRIPT,
    DomAxGrounder,
    LocatorCandidateProvider,
    LocatorFallbackMode,
)
from .targets import DurableTarget


# Bounded settle policy: wait at most about eight seconds for two matching
# network-and-DOM samples after at least three polls.
_POLL_MS = 200
_MIN_POLLS = 3
_STABLE_MATCHES = 2
_MAX_POLLS = 40


class _Locator(Protocol):
    def click(self) -> None: ...

    def count(self) -> int: ...

    def evaluate(self, expression: str, arg: object | None = None): ...

    def fill(self, value: str) -> None: ...

    def element_handle(self) -> _ElementHandle | None: ...

    def filter(self, *, visible: bool | None = None) -> _Locator: ...

    def get_attribute(self, name: str) -> str | None: ...

    def is_editable(self) -> bool: ...

    def is_enabled(self) -> bool: ...

    def is_visible(self) -> bool: ...

    def select_option(self, *, label: str) -> None: ...


class _ElementHandle(Protocol):
    def evaluate(self, expression: str, arg: object | None = None): ...

    def fill(self, value: str) -> None: ...


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


class DomAxBrowserAdapter:
    """Observe through DOM+AX, execute live refs, and replay witnessed targets."""

    def __init__(
        self,
        page: _Page,
        *,
        grounder: DomAxGrounder | None = None,
        locator_fallback: LocatorFallbackMode = "stored_candidates",
        locator_candidate_provider: LocatorCandidateProvider | None = None,
        include_raw_class: bool = False,
    ) -> None:
        if locator_fallback not in {
            "none",
            "stored_candidates",
            "stored_then_llm",
        }:
            raise ValueError(f"unknown locator fallback mode: {locator_fallback!r}")
        if (
            locator_fallback == "stored_then_llm"
            and locator_candidate_provider is None
        ):
            raise ValueError(
                "stored_then_llm requires a locator candidate provider"
            )
        if not isinstance(include_raw_class, bool):
            raise ValueError("include_raw_class must be a boolean")
        self._page = page
        self._grounder = grounder or DomAxGrounder(  # type: ignore[arg-type]
            page,
            include_raw_class=include_raw_class,
        )
        self._locator_fallback = locator_fallback
        self._locator_candidate_provider = locator_candidate_provider
        # Streaming requests never settle, so the callbacks exclude them.
        self._pending_requests: set[object] = set()
        on = getattr(page, "on", None)
        if callable(on):
            on("request", self._request_started)
            on("requestfinished", self._request_finished)
            on("requestfailed", self._request_finished)

    def observe(self) -> Observation:
        """Return the model-readable snapshot plus ref execution metadata."""

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
        """Execute one semantic step, then require a bounded quiet state.

        Live refs return ``None``. Durable replay returns the step pruned to the
        locators confirmed in this capture. The editable-combobox step contains
        one fill and one freshly grounded option click. A settle error means an
        action may already have happened.
        """

        if step.get("op") == "choose_combobox_option":
            return self._execute_choose_combobox_option(step)

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
        """Execute a ref only against the snapshot that created it."""

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
        """Preflight a witnessed target in a fresh capture, then execute it."""

        action = action_from_step(step)
        label = action.label if isinstance(action, SelectOption) else None
        locator, effective_target = self._grounder.resolve_durable_target(
            action.target,
            step.get("op", ""),
            label=label,
            fallback_mode=self._locator_fallback,
            candidate_provider=self._locator_candidate_provider,
        )
        _execute_locator(locator, step.get("op"), step)
        return action_to_step(_action_with_target(action, effective_target))

    def _execute_choose_combobox_option(
        self,
        step: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        """Type, freshly ground one option, click once, then verify its commit."""

        if set(step) != {"op", "target", "label"}:
            raise ValueError("invalid choose_combobox_option action")
        label = step.get("label")
        target = step.get("target")
        if not isinstance(label, str) or not label:
            raise ValueError("choose_combobox_option requires a non-empty label")
        if not isinstance(target, Mapping):
            raise ValueError("choose_combobox_option requires a target mapping")

        effective_target: DurableTarget | None
        if target.get("by") == "ref":
            if set(target) != {"by", "snapshot", "ref"}:
                raise ValueError("invalid snapshot-ref target")
            token = target.get("snapshot")
            ref = target.get("ref")
            if not isinstance(token, str) or not isinstance(ref, str):
                raise ValueError("snapshot-ref target requires string identity")
            field_locator = self._grounder.resolve_ref(
                token,
                ref,
                "choose_combobox_option",
            )
            field_target = self._grounder.durable_target_for_ref(token, ref)
            effective_target = None
        else:
            action = action_from_step(step)
            if not isinstance(action, ChooseComboboxOption):
                raise ValueError("invalid choose_combobox_option action")
            field_locator, field_target = self._grounder.resolve_durable_target(
                action.target,
                "choose_combobox_option",
                fallback_mode=self._locator_fallback,
                candidate_provider=self._locator_candidate_provider,
            )
            effective_target = field_target

        field_handle = field_locator.element_handle()
        if field_handle is None:
            raise ValueError("editable combobox has no live element handle")
        original_value = _editable_value(field_locator)
        try:
            field_locator.fill(label)
            self._wait_until_stable()
            option_locator = self._grounder.resolve_fresh_option(label)
        except Exception as exc:
            try:
                if field_target is None:
                    restore_locator = field_handle
                else:
                    restore_locator, _ = self._grounder.resolve_durable_target(
                        field_target,
                        "choose_combobox_option",
                        fallback_mode="stored_candidates",
                    )
                restore_locator.fill(original_value)
                self._wait_until_stable()
            except Exception as restore_error:
                raise ActionDispatchedError(
                    "combobox preparation failed and its prior value could not "
                    "be restored"
                ) from restore_error
            raise ActionNotCommittedError(
                "combobox option was not uniquely available before click"
            ) from exc

        # Once the option click is attempted, never click it again here. Any
        # settle, fresh-field lookup, or readback uncertainty is post-dispatch.
        try:
            option_locator.click()
            self._wait_until_stable()
            if field_target is None:
                readback_locator = field_handle
                readback_target = None
            else:
                readback_locator, readback_target = (
                    self._grounder.resolve_durable_target(
                        field_target,
                        "choose_combobox_option",
                        fallback_mode=self._locator_fallback,
                        candidate_provider=self._locator_candidate_provider,
                    )
                )
            if not self._combobox_commit_matches(readback_locator, label):
                raise ValueError("editable combobox has no matching commit evidence")
        except Exception as exc:
            raise ActionDispatchedError(
                "combobox option click was dispatched but could not be verified"
            ) from exc

        if effective_target is None:
            return None
        if readback_target is None:
            raise AssertionError("durable combobox replay lost its target")
        return action_to_step(ChooseComboboxOption(readback_target, label))

    def _combobox_commit_matches(
        self,
        field: _Locator | _ElementHandle,
        label: str,
    ) -> bool:
        """Accept exact field, selected-option, hidden-value, or pill evidence."""

        evidence = field.evaluate(
            """(element, expected) => {
                const normalize = value => String(value || '')
                    .replace(/\\s+/g, ' ').trim();
                const wanted = normalize(expected);
                const value = 'value' in element
                    ? normalize(element.value)
                    : normalize(element.textContent);
                const expandedOwner = element.hasAttribute('aria-expanded')
                    ? element
                    : element.closest('[aria-expanded]');
                const expanded = expandedOwner
                    ? expandedOwner.getAttribute('aria-expanded')
                    : null;
                let relatedExact = false;
                let scope = element.parentElement;
                for (let depth = 0; scope && depth < 4 && !relatedExact; depth++) {
                    const candidates = Array.from(scope.querySelectorAll('*')).slice(0, 200);
                    for (const candidate of candidates) {
                        if (candidate === element || candidate.contains(element)) continue;
                        if (candidate.closest('[role="listbox"], [role="option"]')) continue;
                        const hiddenInput = candidate.matches('input[type="hidden"]');
                        if (!hiddenInput) {
                            const rect = candidate.getBoundingClientRect();
                            const style = getComputedStyle(candidate);
                            if (rect.width === 0 || rect.height === 0 ||
                                style.display === 'none' || style.visibility === 'hidden' ||
                                Number(style.opacity) < 0.05) continue;
                        }
                        const values = [
                            candidate.textContent,
                            candidate.getAttribute('aria-label'),
                            candidate.getAttribute('title'),
                            candidate.getAttribute('data-value'),
                        ];
                        if (hiddenInput) {
                            values.push(candidate.value);
                        }
                        if (values.some(item => normalize(item) === wanted)) {
                            relatedExact = true;
                            break;
                        }
                    }
                    scope = scope.parentElement;
                }
                return {value, expanded, relatedExact};
            }""",
            label,
        )
        if not isinstance(evidence, Mapping):
            return False
        wanted = _normalize_text(label)
        value_matches = _normalize_text(evidence.get("value")) == wanted
        related_matches = evidence.get("relatedExact") is True
        exact_options = self._page.get_by_role(
            "option",
            name=label,
            exact=True,
        )
        visible_options = exact_options.filter(visible=True)
        selected_matches = (
            visible_options.count() == 1
            and visible_options.get_attribute("aria-selected") == "true"
        )
        menu_closed = (
            evidence.get("expanded") == "false"
            or visible_options.count() == 0
        )
        return related_matches or selected_matches or (value_matches and menu_closed)

    def _wait_until_stable(self) -> None:
        """Wait for both non-streaming requests and DOM revisions to go quiet."""

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
    """Replace an action target with the candidates confirmed during replay."""

    if isinstance(action, Click):
        return Click(target)
    if isinstance(action, Fill):
        return Fill(target, action.value)
    if isinstance(action, SelectOption):
        return SelectOption(target, action.label)
    return ChooseComboboxOption(target, action.label)


def _editable_value(locator: _Locator | _ElementHandle) -> str:
    """Read the current text from a native or contenteditable combobox."""

    value = locator.evaluate(
        """element => {
            if ('value' in element) return String(element.value || '');
            if (element.isContentEditable) return element.textContent || '';
            return '';
        }"""
    )
    return value if isinstance(value, str) else ""


def _normalize_text(value: object) -> str:
    """Compare browser readback without treating harmless whitespace as drift."""

    return " ".join(str(value or "").split())
