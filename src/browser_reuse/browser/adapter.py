"""Site-neutral browser adapter over DOM+AX snapshot-local refs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from browser_reuse.core import Observation
from browser_reuse.interfaces import ActionDispatchedError, ActionNotCommittedError

from .actions import (
    Appears,
    BrowserAction,
    ChooseComboboxOption,
    Click,
    Fill,
    SelectOption,
    SetChecked,
    UrlIs,
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
_READBACK_STABLE_MATCHES = 2
_READBACK_MAX_POLLS = 10
_CLICK_READBACK_MAX_POLLS = 40


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
                "readback_facts": [
                    {"role": role, "name": name}
                    for role, name in snapshot.readback_facts
                ],
            }
        )

    def execute(self, step: Mapping[str, object]) -> Mapping[str, object] | None:
        """Execute one semantic step and confirm its strongest local evidence.

        Live refs return ``None``. Durable replay returns the step pruned to the
        locators confirmed in this capture. A custom-combobox step prepares its
        field and performs one freshly grounded option click. Fill and option
        actions use field-owned readback; generic clicks use bounded page quiet.
        """

        if step.get("op") == "choose_combobox_option":
            return self._execute_choose_combobox_option(step)
        if step.get("op") == "set_checked":
            return self._execute_set_checked(step)
        readback = step.get("readback")
        if (
            step.get("op") == "click"
            and isinstance(readback, Mapping)
            and readback.get("kind") == "url_is"
        ):
            return self._execute_url_click(step)

        target = step.get("target")
        effective_step: Mapping[str, object] | None = None
        readback_target: DurableTarget | None = None
        click_readback: Appears | None = None
        if isinstance(target, Mapping) and target.get("by") == "ref":
            locator, readback_target, dispatched = self._execute_ref_step(
                step, target
            )
        else:
            action = action_from_step(step)
            if isinstance(action, Click):
                click_readback = action.readback
                if (
                    click_readback is not None
                    and self._grounder.semantic_fact_count(
                        click_readback.role,
                        click_readback.name,
                    )
                    != 0
                ):
                    raise ValueError(
                        "click readback already holds before dispatch"
                    )
            locator, effective_step, dispatched = self._execute_durable_step(step)
            readback_target = action_from_step(effective_step).target

        operation = step.get("op")
        if operation in {"fill", "select_option"}:
            if not dispatched:
                return effective_step
            readback_locator = locator
            confirmed_target = readback_target

            def commit_matches() -> bool:
                nonlocal confirmed_target, readback_locator
                if self._typed_readback_matches(
                    readback_locator, operation, step
                ):
                    return True
                if confirmed_target is None:
                    return False
                label = (
                    step.get("label")
                    if operation == "select_option"
                    else None
                )
                try:
                    fresh, confirmed = self._grounder.resolve_durable_target(
                        confirmed_target,
                        str(operation),
                        label=label if isinstance(label, str) else None,
                        fallback_mode="stored_candidates",
                    )
                except Exception:
                    return False
                readback_locator = fresh
                confirmed_target = confirmed
                return self._typed_readback_matches(
                    readback_locator, operation, step
                )

            if self._wait_for_readback(commit_matches):
                if effective_step is not None and confirmed_target is not None:
                    return action_to_step(
                        _action_with_target(
                            action_from_step(effective_step),
                            confirmed_target,
                        )
                    )
                return effective_step
            raise ActionDispatchedError(
                f"{operation} was dispatched but field readback was unresolved"
            )

        if click_readback is not None:
            if self._wait_for_readback(
                lambda: self._grounder.semantic_fact_count(
                    click_readback.role,
                    click_readback.name,
                )
                == 1,
                max_polls=_CLICK_READBACK_MAX_POLLS,
            ):
                return effective_step
            raise ActionDispatchedError(
                "click was dispatched but semantic readback was unresolved"
            )

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
    ) -> tuple[_Locator, DurableTarget | None, bool]:
        """Resolve a ref, then dispatch only when its desired state is absent."""

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
        locator = self._resolve_source_ref(
            token,
            ref,
            str(operation),
            label=label if isinstance(label, str) else None,
        )
        try:
            durable_target = self._grounder.durable_target_for_ref(token, ref)
        except Exception as exc:
            raise ActionNotCommittedError(str(exc)) from exc
        if operation in {"fill", "select_option"} and self._typed_readback_matches(
            locator, operation, step
        ):
            return locator, durable_target, False
        _execute_locator(locator, operation, step)
        return locator, durable_target, True

    def _resolve_source_ref(
        self,
        token: str,
        ref: str,
        operation: str,
        *,
        label: str | None = None,
    ) -> _Locator:
        """Classify every live-ref rejection before dispatch as recoverable."""

        try:
            return self._grounder.resolve_ref(
                token,
                ref,
                operation,
                label=label,
            )
        # ``resolve_ref`` is a read-only preflight. Even a Playwright/CDP
        # protocol failure here happened before ``_execute_locator`` and is
        # therefore safe for the Agent to diagnose from a fresh observation.
        except Exception as exc:
            raise ActionNotCommittedError(str(exc)) from exc

    def _execute_durable_step(
        self,
        step: Mapping[str, object],
    ) -> tuple[_Locator, Mapping[str, object], bool]:
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
        effective_step = action_to_step(
            _action_with_target(action, effective_target)
        )
        operation = step.get("op")
        if operation in {"fill", "select_option"} and self._typed_readback_matches(
            locator, operation, step
        ):
            return locator, effective_step, False
        _execute_locator(locator, operation, step)
        return locator, effective_step, True

    def _execute_choose_combobox_option(
        self,
        step: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        """Prepare a field, freshly ground one option, and verify one click."""

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
            field_locator = self._resolve_source_ref(
                token,
                ref,
                "choose_combobox_option",
            )
            try:
                field_target = self._grounder.durable_target_for_ref(token, ref)
            except ValueError as exc:
                raise ActionNotCommittedError(str(exc)) from exc
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
            raise ValueError("combobox has no live element handle")
        editable = field_locator.is_editable()
        original_value = _editable_value(field_locator) if editable else ""
        try:
            if editable:
                field_locator.fill(label)
            elif field_locator.get_attribute("aria-expanded") != "true":
                field_locator.click()
            option_locator = self._wait_for_fresh_option(label)
        except Exception as exc:
            if not editable:
                raise ActionNotCommittedError(
                    "combobox option was not uniquely available before click"
                ) from exc
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
                restored = self._wait_for_readback(
                    lambda: _editable_value(restore_locator) == original_value
                )
                if not restored:
                    raise ValueError("combobox value could not be restored")
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
        option_locator.click()
        readback_target = effective_target

        def commit_matches() -> bool:
            nonlocal readback_target
            try:
                if self._combobox_commit_matches(field_handle, label):
                    return True
            except Exception:
                pass
            if field_target is None:
                return False
            readback_locator, confirmed_target = (
                self._grounder.resolve_durable_target(
                    field_target,
                    "choose_combobox_option",
                    fallback_mode=self._locator_fallback,
                    candidate_provider=self._locator_candidate_provider,
                )
            )
            readback_target = confirmed_target
            return self._combobox_commit_matches(readback_locator, label)

        if not self._wait_for_readback(commit_matches):
            raise ActionDispatchedError(
                "combobox option click was dispatched but could not be verified"
            )

        if effective_target is None:
            return None
        if readback_target is None:
            raise AssertionError("durable combobox replay lost its target")
        return action_to_step(ChooseComboboxOption(readback_target, label))

    def _execute_set_checked(
        self,
        step: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Reach one witnessed choice state with at most one click."""

        action = action_from_step(step)
        if not isinstance(action, SetChecked):
            raise ValueError("invalid set_checked action")
        locator, target = self._grounder.resolve_durable_target(
            action.target,
            "set_checked",
            fallback_mode=self._locator_fallback,
            candidate_provider=self._locator_candidate_provider,
        )
        kind, current = _checked_control_state(locator)
        if kind is None or current is None:
            raise ValueError("set_checked target has no exact boolean state")
        if kind == "radio" and not action.checked:
            raise ValueError("a radio target cannot be set to unchecked")
        if current == action.checked:
            return action_to_step(SetChecked(target, action.checked))

        # This is the only commit point. Readback may re-resolve the witnessed
        # node, but it never dispatches a second click.
        locator.click()
        readback_locator = locator
        readback_target = target

        def commit_matches() -> bool:
            nonlocal readback_locator, readback_target
            try:
                live_kind, live_state = _checked_control_state(readback_locator)
                if live_kind == kind and live_state == action.checked:
                    return True
            except Exception:
                pass
            try:
                fresh, confirmed = self._grounder.resolve_durable_target(
                    target,
                    "set_checked",
                    fallback_mode="stored_candidates",
                )
            except Exception:
                return False
            readback_locator = fresh
            readback_target = confirmed
            live_kind, live_state = _checked_control_state(fresh)
            return live_kind == kind and live_state == action.checked

        if not self._wait_for_readback(commit_matches):
            raise ActionDispatchedError(
                "set_checked was dispatched but target readback was unresolved"
            )
        return action_to_step(SetChecked(readback_target, action.checked))

    def _execute_url_click(
        self,
        step: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Follow one same-origin link to its exact witnessed destination."""

        action = action_from_step(step)
        if not isinstance(action, Click) or not isinstance(
            action.readback, UrlIs
        ):
            raise ValueError("invalid url-readback click action")
        locator, target = self._grounder.resolve_durable_target(
            action.target,
            "click",
            fallback_mode=self._locator_fallback,
            candidate_provider=self._locator_candidate_provider,
        )
        expected = action.readback.relative_url
        if _same_origin_link_destination(locator, self._page.url) != expected:
            raise ValueError("link href does not match its exact URL readback")
        if _relative_http_url(self._page.url) == expected:
            return action_to_step(Click(target, action.readback))

        locator.click()
        if not self._wait_for_readback(
            lambda: _relative_http_url(self._page.url) == expected,
            max_polls=_CLICK_READBACK_MAX_POLLS,
        ):
            raise ActionDispatchedError(
                "link click was dispatched but exact URL readback was unresolved"
            )
        return action_to_step(Click(target, action.readback))

    def _wait_for_fresh_option(self, label: str) -> _Locator:
        """Wait only for one freshly grounded exact option, not global quiet."""

        last_error: Exception | None = None
        for _ in range(_READBACK_MAX_POLLS):
            self._page.wait_for_timeout(_POLL_MS)
            try:
                return self._grounder.resolve_fresh_option(label)
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise ValueError("combobox option was not available")

    def _wait_for_readback(
        self,
        matches: Callable[[], bool],
        *,
        max_polls: int = _READBACK_MAX_POLLS,
    ) -> bool:
        """Require a local positive signal to persist for two render turns."""

        stable_matches = 0
        for _ in range(max_polls):
            self._page.wait_for_timeout(_POLL_MS)
            try:
                matched = matches()
            except Exception:
                matched = False
            stable_matches = stable_matches + 1 if matched else 0
            if stable_matches >= _READBACK_STABLE_MATCHES:
                return True
        return False

    def _typed_readback_matches(
        self,
        locator: _Locator,
        operation: object,
        step: Mapping[str, object],
    ) -> bool:
        """Check a field's own value or selected label after one dispatch."""

        try:
            if operation == "fill":
                expected = step.get("value")
                return (
                    isinstance(expected, str)
                    and _editable_value(locator) == expected
                )
            if operation == "select_option":
                expected = step.get("label")
                if not isinstance(expected, str):
                    return False
                selected = locator.evaluate(
                    """element => Array.from(element.selectedOptions || [])
                        .map(option => String(option.label || option.textContent || '')
                            .replace(/\\s+/g, ' ').trim())"""
                )
                return isinstance(selected, list) and selected == [
                    _normalize_text(expected)
                ]
            return False
        except Exception:
            # Dispatch already happened. A detached or unreadable field is not
            # evidence of failure or success, so the caller reports unresolved.
            return False

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
        return Click(target, action.readback)
    if isinstance(action, Fill):
        return Fill(target, action.value)
    if isinstance(action, SelectOption):
        return SelectOption(target, action.label)
    if isinstance(action, ChooseComboboxOption):
        return ChooseComboboxOption(target, action.label)
    return SetChecked(target, action.checked)


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


def _checked_control_state(
    locator: _Locator,
) -> tuple[str | None, bool | None]:
    """Read only the exact target's native or ARIA checked state."""

    value = locator.evaluate(
        """element => {
            const tag = element.tagName.toLowerCase();
            const type = String(element.getAttribute('type') || '').toLowerCase();
            if (tag === 'input' && (type === 'checkbox' || type === 'radio')) {
                return {kind: type, checked: Boolean(element.checked)};
            }
            const role = String(element.getAttribute('role') || '').toLowerCase();
            if (role !== 'checkbox' && role !== 'radio') return null;
            const checked = element.getAttribute('aria-checked');
            if (checked !== 'true' && checked !== 'false') {
                return {kind: role, checked: null};
            }
            return {kind: role, checked: checked === 'true'};
        }"""
    )
    if not isinstance(value, Mapping):
        return None, None
    kind = value.get("kind")
    checked = value.get("checked")
    if kind not in {"checkbox", "radio"} or not isinstance(checked, bool):
        return None, None
    return str(kind), checked


def _relative_http_url(value: str) -> str | None:
    """Return one exact path and query for an HTTP URL without a fragment."""

    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.fragment
    ):
        return None
    return urlunsplit(("", "", parsed.path or "/", parsed.query, ""))


def _same_origin_link_destination(
    locator: _Locator,
    current_url: str,
) -> str | None:
    """Read a same-tab anchor destination without using it as identity."""

    value = locator.evaluate(
        """element => ({
            tag: element.tagName.toLowerCase(),
            href: element.href || '',
            target: element.getAttribute('target') || '',
            download: element.hasAttribute('download'),
        })"""
    )
    if not isinstance(value, Mapping):
        return None
    target = value.get("target")
    if (
        value.get("tag") != "a"
        or value.get("download") is True
        or target not in {"", "_self"}
        or not isinstance(value.get("href"), str)
    ):
        return None
    current = urlsplit(current_url)
    destination = urlsplit(str(value["href"]))
    if (
        current.scheme not in {"http", "https"}
        or destination.scheme not in {"http", "https"}
        or not current.netloc
        or current.scheme != destination.scheme
        or current.netloc != destination.netloc
        or destination.fragment
    ):
        return None
    return urlunsplit(
        ("", "", destination.path or "/", destination.query, "")
    )
