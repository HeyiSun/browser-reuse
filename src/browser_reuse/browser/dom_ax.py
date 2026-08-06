"""Join Chromium DOM and accessibility trees into executable browser refs."""

from __future__ import annotations

import json
import re
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .actions import Click, action_to_step
from .targets import (
    BrowserLocator,
    ContextFact,
    CssLocator,
    DurableTarget,
    ElementWitness,
    RoleLocator,
    XPathLocator,
    looks_generated_id,
)


# A random marker connects a CDP backend node to a Playwright locator for one
# snapshot only. The revision observer ignores these private marker mutations.
REF_ATTRIBUTE_PREFIX = "data-browser-reuse-ref-"
DOM_REVISION_SCRIPT = f"""() => {{
    const key = '__browserReuseDomRevisionV2';
    if (!globalThis[key]) {{
        const state = {{version: 0, roots: new WeakSet()}};
        const observer = new MutationObserver((mutations) => {{
            if (mutations.some(mutation => !(
                mutation.type === 'attributes' &&
                (mutation.attributeName || '').startsWith('{REF_ATTRIBUTE_PREFIX}')
            ))) {{
                state.version += 1;
            }}
            scan(false);
        }});
        const observe = (root, initial) => {{
            if (!root || state.roots.has(root)) return;
            state.roots.add(root);
            observer.observe(root, {{
                attributes: true,
                childList: true,
                characterData: true,
                subtree: true,
            }});
            if (!initial) state.version += 1;
        }};
        const scanRoot = (root, initial) => {{
            observe(root, initial);
            if (!root) return;
            for (const element of root.querySelectorAll('*')) {{
                if (element.shadowRoot) scanRoot(element.shadowRoot, initial);
            }}
        }};
        const scan = (initial) => scanRoot(document.documentElement, initial);
        state.scan = scan;
        globalThis[key] = state;
        scan(true);
    }}
    globalThis[key].scan(false);
    return {{ready: document.readyState, version: globalThis[key].version}};
}}"""
# The current action vocabulary is intentionally limited to roles proven by
# existing tasks. Unsupported controls remain visible but receive no ref.
_CLICK_ROLES = frozenset(
    {
        "button",
        "checkbox",
        "link",
        "menuitem",
        "option",
        "radio",
        "switch",
        "tab",
    }
)
_FILL_ROLES = frozenset({"searchbox", "spinbutton", "textbox"})
_STRUCTURAL_ROLES = frozenset(
    {"generic", "none", "inlinetextbox", "rootwebarea"}
)
# Attribute order is evidence priority, not a complete DOM attribute list.
_TEST_ATTRIBUTES = (
    "data-testid",
    "data-test-id",
    "data-test",
    "data-qa",
    "data-cy",
)
_IDENTITY_ATTRIBUTES = (
    *_TEST_ATTRIBUTES,
    "name",
    "aria-label",
    "placeholder",
    "type",
)
_CONTEXT_ANCESTOR_ROLES = frozenset(
    {
        "article",
        "dialog",
        "form",
        "group",
        "listitem",
        "region",
        "row",
        "table",
        "tabpanel",
    }
)
# These budgets fail closed before an oversized snapshot reaches the model.
_MAX_SELECT_OPTIONS = 40
_MAX_CONTROLS = 200
_MAX_SNAPSHOT_BYTES = 64_000
# Secret hints only control conservative redaction; they never identify nodes.
_SECRET_AUTOCOMPLETE = frozenset(
    {
        "current-password",
        "new-password",
        "one-time-code",
        "cc-name",
        "cc-number",
        "cc-csc",
        "cc-exp",
        "cc-exp-month",
        "cc-exp-year",
    }
)
_SECRET_TEXT = re.compile(
    r"(?:password|passcode|one[- ]?time|otp|verification code|"
    r"api[-_ ]?(?:key|token)|secret|access[-_ ]?token|social security|ssn|"
    r"card number|credit card|cvv|cvc)",
    re.IGNORECASE,
)


class _CdpSession(Protocol):
    def send(
        self,
        method: str,
        params: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]: ...

    def detach(self) -> None: ...


class _BrowserContext(Protocol):
    def new_cdp_session(self, page: object) -> _CdpSession: ...


class _Locator(Protocol):
    def count(self) -> int: ...

    def evaluate(self, expression: str, arg: object | None = None): ...

    def evaluate_all(self, expression: str, arg: object | None = None): ...

    def get_attribute(self, name: str) -> str | None: ...

    def input_value(self) -> str: ...

    def is_editable(self) -> bool: ...

    def is_enabled(self) -> bool: ...

    def is_visible(self) -> bool: ...


class _Page(Protocol):
    context: _BrowserContext
    frames: Sequence[object]

    def evaluate(self, expression: str): ...

    def get_by_role(
        self,
        role: str,
        *,
        name: str,
        exact: bool,
    ) -> _Locator: ...

    def locator(self, selector: str) -> _Locator: ...

    def wait_for_timeout(self, timeout: float) -> None: ...


@dataclass(frozen=True)
class BrowserSnapshot:
    """One coherent model view plus its executable ref table and diagnostics."""

    text: str
    controls: Mapping[str, Mapping[str, object]]
    token: str
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class _DomNode:
    """DOM facts indexed by Chromium backend node identity."""

    tag: str
    attributes: Mapping[str, str]
    closed_shadow: bool
    in_shadow: bool
    absolute_xpath: str | None


@dataclass(frozen=True)
class _ContextEvidence:
    """A named semantic ancestor or heading attached to its DOM evidence."""

    fact: ContextFact
    backend_id: int
    tag: str
    attributes: Mapping[str, str]


@dataclass(frozen=True)
class _ActionableNode:
    """Joined DOM+AX facts for a control the current runtime can execute."""

    ref: str
    backend_id: int
    tag: str
    attributes: Mapping[str, str]
    role: str
    name: str
    operation: str
    contexts: tuple[_ContextEvidence, ...]
    in_shadow: bool
    absolute_xpath: str | None


@dataclass(frozen=True)
class _WitnessOption:
    """One admitted fact that may help make a witness unique."""

    fact_id: str
    kind: str
    key: str
    value: str
    context: ContextFact | None = None

    def public(self) -> Mapping[str, str]:
        value = {
            "fact_id": self.fact_id,
            "kind": self.kind,
            "key": self.key,
            "value": self.value,
        }
        if self.context is not None:
            value["relation"] = self.context.relation
            value["role"] = self.context.role
        return value


WitnessFactSelector = Callable[
    [tuple[Mapping[str, str], ...]],
    Sequence[str],
]


@dataclass(frozen=True)
class _RefBinding:
    """Capture-time identity used to reject semantic drift before execution."""

    marker: _Marker
    operation: str
    backend_id: int
    tag: str
    role: str
    name: str
    state: tuple[tuple[str, object], ...]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class _Marker:
    """A temporary DOM attribute/value pair bound to one snapshot."""

    attribute: str
    value: str


class _TransientCaptureError(RuntimeError):
    """A page race that permits one whole-capture retry."""


class DomAxGrounder:
    """Capture one main-document DOM+AX view and bind refs to exact live nodes."""

    def __init__(
        self,
        page: _Page,
        *,
        witness_fact_selector: WitnessFactSelector | None = None,
    ) -> None:
        self._page = page
        self._witness_fact_selector = witness_fact_selector
        # All fields below belong to the latest capture and are replaced as a unit.
        self._token: str | None = None
        self._marker_attribute: str | None = None
        self._loader_id: str | None = None
        self._bindings: dict[str, _RefBinding] = {}
        self._actionable_nodes: tuple[_ActionableNode, ...] = ()

    def capture(self) -> BrowserSnapshot:
        """Capture main-document and open-shadow DOM+AX with one transient retry."""

        return self._capture(include_targets=True)

    def _capture(self, *, include_targets: bool) -> BrowserSnapshot:
        """Retry only when navigation or mutation tears the whole capture."""

        for attempt in range(2):
            try:
                return self._capture_once(include_targets=include_targets)
            except _TransientCaptureError:
                if attempt == 1:
                    raise
                self._page.wait_for_timeout(200)
        raise AssertionError("unreachable capture retry state")

    def _capture_once(self, *, include_targets: bool) -> BrowserSnapshot:
        """Build one loader- and revision-consistent snapshot."""

        started = time.perf_counter()
        self._remove_previous_markers()
        revision_before = _dom_revision(self._page)
        token = secrets.token_hex(8)
        marker_attribute = f"{REF_ATTRIBUTE_PREFIX}{token}"
        session = self._page.context.new_cdp_session(self._page)
        try:
            session.send("Page.enable")
            session.send("DOM.enable")
            session.send("Accessibility.enable")
            frame_id, loader_before = _main_frame_identity(session)
            dom_result = session.send(
                "DOM.getDocument",
                {"depth": -1, "pierce": True},
            )
            ax_result = session.send(
                "Accessibility.getFullAXTree",
                {"frameId": frame_id},
            )
            _, loader_after = _main_frame_identity(session)
            if loader_before != loader_after:
                raise _TransientCaptureError(
                    "document changed while capturing DOM and AX"
                )

            dom_root = dom_result.get("root")
            ax_nodes = ax_result.get("nodes")
            if not isinstance(dom_root, Mapping) or not isinstance(ax_nodes, list):
                raise RuntimeError("CDP returned an invalid DOM or AX snapshot")

            dom_nodes, closed_shadow_roots = _index_main_document(dom_root)
            diagnostics: dict[str, object] = {
                "dom_nodes": len(dom_nodes),
                "ax_nodes": len(ax_nodes),
                "joined_ax_nodes": 0,
                "unresolved_ax_nodes": 0,
                "actionable_refs": 0,
                "durable_targets": 0,
                "unsupported_frames": max(0, len(self._page.frames) - 1),
                "unsupported_closed_shadow_roots": closed_shadow_roots,
            }
            controls: dict[str, dict[str, object]] = {}
            ax_by_id = {
                str(node.get("nodeId")): node
                for node in ax_nodes
                if isinstance(node, Mapping) and node.get("nodeId") is not None
            }
            child_ids = {
                str(child_id)
                for node in ax_by_id.values()
                for child_id in _sequence(node.get("childIds"))
            }
            roots = [node_id for node_id in ax_by_id if node_id not in child_ids]
            if not roots and ax_by_id:
                roots = [next(iter(ax_by_id))]

            lines: list[str] = []
            visited: set[str] = set()
            actionable_nodes: list[_ActionableNode] = []
            # A one-item list shares the latest heading across recursive DFS calls.
            heading_state: list[_ContextEvidence | None] = [None]
            for root_id in roots:
                self._render_ax_node(
                    root_id,
                    ax_by_id,
                    dom_nodes,
                    session,
                    marker_attribute,
                    controls,
                    diagnostics,
                    lines,
                    actionable_nodes,
                    heading_state,
                    ancestor_contexts=(),
                    depth=0,
                    visited=visited,
                )

            _, loader_final = _main_frame_identity(session)
            if loader_before != loader_final:
                raise _TransientCaptureError(
                    "document changed while publishing browser refs"
                )
            if _dom_revision(self._page) != revision_before:
                raise _TransientCaptureError(
                    "document mutated while capturing DOM and AX"
                )
        except Exception:
            _remove_page_markers(self._page, marker_attribute)
            raise
        finally:
            session.detach()

        frozen_actionable_nodes = tuple(actionable_nodes)
        self._actionable_nodes = frozen_actionable_nodes
        if include_targets:
            try:
                self._attach_durable_targets(
                    controls,
                    frozen_actionable_nodes,
                )
                verification = self._page.context.new_cdp_session(self._page)
                try:
                    verification.send("Page.enable")
                    _, loader_after_targets = _main_frame_identity(verification)
                finally:
                    verification.detach()
                if loader_after_targets != loader_before:
                    raise _TransientCaptureError(
                        "document changed while compiling durable targets"
                    )
            except Exception:
                _remove_page_markers(self._page, marker_attribute)
                raise

        diagnostics["actionable_refs"] = len(controls)
        diagnostics["durable_targets"] = sum(
            1 for control in controls.values() if "target" in control
        )
        diagnostics["capture_ms"] = round(
            (time.perf_counter() - started) * 1_000,
            3,
        )
        text = "\n".join(lines)
        diagnostics["snapshot_bytes"] = len(text.encode("utf-8"))
        if len(controls) > _MAX_CONTROLS:
            _remove_page_markers(self._page, marker_attribute)
            raise RuntimeError("browser observation exceeds the actionable control budget")
        if diagnostics["snapshot_bytes"] > _MAX_SNAPSHOT_BYTES:
            _remove_page_markers(self._page, marker_attribute)
            raise RuntimeError("browser observation exceeds the snapshot byte budget")

        bindings: dict[str, _RefBinding] = {}
        for ref, control in controls.items():
            # Private staging fields build ref bindings, then leave public output.
            backend_id = control.pop("_backend_id", None)
            if not isinstance(backend_id, int):
                raise RuntimeError("published browser ref lost its backend identity")
            bindings[ref] = _RefBinding(
                marker=_Marker(marker_attribute, ref),
                operation=str(control["op"]),
                backend_id=backend_id,
                tag=str(control.pop("_tag")),
                role=str(control.pop("_role")),
                name=str(control.pop("_name")),
                state=tuple(control.pop("_state")),
                labels=tuple(control.get("labels", ())),
            )
        self._token = token
        self._marker_attribute = marker_attribute
        self._loader_id = loader_before
        self._bindings = bindings
        return BrowserSnapshot(
            text=text,
            controls=controls,
            token=token,
            diagnostics=diagnostics,
        )

    def resolve_ref(
        self,
        token: str,
        ref: str,
        operation: str,
        *,
        label: str | None = None,
    ) -> _Locator:
        """Resolve a ref only if its loader, DOM node, semantics, and options match."""

        if (
            token != self._token
            or self._loader_id is None
        ):
            raise ValueError("browser ref belongs to a stale observation")
        binding = self._bindings.get(ref)
        if binding is None or binding.operation != operation:
            raise ValueError("browser ref/action is not present in the observation")

        session = self._page.context.new_cdp_session(self._page)
        try:
            session.send("DOM.enable")
            session.send("Accessibility.enable")
            _, loader_id = _main_frame_identity(session)
            dom_result = session.send(
                "DOM.getDocument",
                {"depth": -1, "pierce": True},
            )
            dom_root = dom_result.get("root")
            live_backend_ids = (
                _backend_ids_with_marker(dom_root, binding.marker)
                if isinstance(dom_root, Mapping)
                else ()
            )
            live_dom_nodes = (
                _index_main_document(dom_root)[0]
                if isinstance(dom_root, Mapping)
                else {}
            )
            live_semantics = _backend_semantics(session, binding.backend_id)
        finally:
            session.detach()
        if loader_id != self._loader_id:
            raise ValueError("browser ref belongs to a replaced document")
        if live_backend_ids != (binding.backend_id,):
            raise ValueError("browser ref no longer resolves to its live node")
        live_dom = live_dom_nodes.get(binding.backend_id)
        if live_dom is None or live_dom.tag != binding.tag:
            raise ValueError("browser ref DOM identity changed")
        if live_semantics != (binding.role, binding.name, binding.state):
            raise ValueError("browser ref semantic identity changed")

        locator = self._page.locator(_marker_selector(binding.marker))
        if locator.count() != 1:
            raise ValueError("browser ref no longer resolves to its live node")
        if not locator.is_visible() or not locator.is_enabled():
            raise ValueError("browser ref is no longer actionable")
        if operation == "fill" and not locator.is_editable():
            raise ValueError("browser ref is no longer editable")
        if operation == "select_option":
            labels = _select_labels(locator)
            if labels != binding.labels or label is None or labels.count(label) != 1:
                raise ValueError("browser select options changed after observation")
        return locator

    def _render_ax_node(
        self,
        node_id: str,
        ax_by_id: Mapping[str, Mapping[str, object]],
        dom_nodes: Mapping[int, _DomNode],
        session: _CdpSession,
        marker_attribute: str,
        controls: dict[str, dict[str, object]],
        diagnostics: dict[str, object],
        lines: list[str],
        actionable_nodes: list[_ActionableNode],
        heading_state: list[_ContextEvidence | None],
        *,
        ancestor_contexts: tuple[_ContextEvidence, ...],
        depth: int,
        visited: set[str],
    ) -> None:
        """Render one AX subtree while joining executable nodes to DOM evidence."""

        if node_id in visited:
            return
        visited.add(node_id)
        node = ax_by_id.get(node_id)
        if node is None:
            return

        role = _ax_text(node.get("role")).lower()
        name = _ax_text(node.get("name"))
        backend_id = node.get("backendDOMNodeId")
        dom_node = (
            dom_nodes.get(backend_id)
            if isinstance(backend_id, int) and not isinstance(backend_id, bool)
            else None
        )
        if dom_node is not None and dom_node.tag == "option":
            return
        if backend_id is not None:
            if dom_node is None:
                diagnostics["unresolved_ax_nodes"] = (
                    int(diagnostics["unresolved_ax_nodes"]) + 1
                )
            else:
                diagnostics["joined_ax_nodes"] = (
                    int(diagnostics["joined_ax_nodes"]) + 1
                )

        nearest_heading = heading_state[0]
        contexts = tuple(reversed(ancestor_contexts))
        if nearest_heading is not None:
            contexts = (*contexts, nearest_heading)

        ref: str | None = None
        display_name = name
        if dom_node is not None and not _ax_ignored(node):
            published = self._publish_control(
                node,
                dom_node,
                int(backend_id),
                session,
                marker_attribute,
                controls,
                actionable_nodes,
                contexts,
            )
            if published is not None:
                ref, display_name = published

        current_context: _ContextEvidence | None = None
        if (
            dom_node is not None
            and isinstance(backend_id, int)
            and name
            and not _ax_ignored(node)
        ):
            if role == "heading":
                heading_state[0] = _ContextEvidence(
                    ContextFact("nearest_heading", "heading", name),
                    backend_id,
                    dom_node.tag,
                    dom_node.attributes,
                )
            if role in _CONTEXT_ANCESTOR_ROLES:
                current_context = _ContextEvidence(
                    ContextFact("ancestor", role, name),
                    backend_id,
                    dom_node.tag,
                    dom_node.attributes,
                )

        child_ids = [str(value) for value in _sequence(node.get("childIds"))]
        if dom_node is not None and _is_secret(dom_node.attributes, name):
            child_ids = []
        show = bool(
            role and (display_name or ref or role not in _STRUCTURAL_ROLES)
        )
        child_depth = depth + 1 if show else depth
        if show:
            line = "  " * depth + f"- {role}"
            if display_name:
                line += f" {json.dumps(display_name, ensure_ascii=False)}"
            for state in ("checked", "selected", "expanded", "disabled"):
                value = _ax_property(node, state)
                if value is True:
                    line += f" [{state}]"
                elif state == "checked" and value == "mixed":
                    line += " [checked=mixed]"
            if role == "button" and any(
                _ax_property(node, state) in (True, "true", 1)
                for state in ("pressed", "selected", "checked")
            ):
                line += " [active]"
            if ref is not None:
                line += f" [ref={ref}]"
            lines.append(line)

        for child_id in child_ids:
            self._render_ax_node(
                child_id,
                ax_by_id,
                dom_nodes,
                session,
                marker_attribute,
                controls,
                diagnostics,
                lines,
                actionable_nodes,
                heading_state,
                ancestor_contexts=(
                    (*ancestor_contexts, current_context)
                    if current_context is not None
                    else ancestor_contexts
                ),
                depth=child_depth,
                visited=visited,
            )

    def _publish_control(
        self,
        ax_node: Mapping[str, object],
        dom_node: _DomNode,
        backend_id: int,
        session: _CdpSession,
        marker_attribute: str,
        controls: dict[str, dict[str, object]],
        actionable_nodes: list[_ActionableNode],
        contexts: tuple[_ContextEvidence, ...],
    ) -> tuple[str, str] | None:
        """Publish one safe live ref, or leave the control non-executable."""

        role = _ax_text(ax_node.get("role")).lower()
        name = _ax_text(ax_node.get("name"))
        operation = _operation(role, dom_node.tag)
        if (
            operation is None
            or dom_node.closed_shadow
            or (
                dom_node.tag == "input"
                and dom_node.attributes.get("type", "").casefold() == "file"
            )
        ):
            return None

        secret = _is_secret(dom_node.attributes, name)
        ref = f"e{len(controls)}"
        marker = _Marker(marker_attribute, ref)
        if not _set_marker(session, backend_id, marker):
            return None
        locator = self._page.locator(_marker_selector(marker))
        if (
            locator.count() != 1
            or not locator.is_visible()
            or not locator.is_enabled()
            or (operation == "fill" and not locator.is_editable())
        ):
            _remove_marker(session, backend_id, marker)
            return None

        public_name = (
            "" if secret and _name_contains_live_value(locator, name) else name
        )
        control: dict[str, object] = {"op": operation, "name": public_name}
        public_state = _public_dom_state(dom_node.attributes)
        if public_state:
            control["state"] = public_state
        if operation == "fill" and not secret:
            control["value"] = locator.input_value()
        elif operation == "select_option":
            if secret:
                _remove_marker(session, backend_id, marker)
                return None
            state = locator.evaluate(
                """element => ({
                    value: element.selectedOptions[0]?.textContent?.trim() || '',
                    labels: Array.from(element.options)
                        .filter(option =>
                            !option.disabled && !option.hidden && option.value
                        )
                        .map(option => option.textContent.trim())
                })"""
            )
            if not isinstance(state, Mapping):
                _remove_marker(session, backend_id, marker)
                return None
            labels = tuple(str(label) for label in _sequence(state.get("labels")))
            if (
                not labels
                or len(labels) > _MAX_SELECT_OPTIONS
                or len(labels) != len(set(labels))
            ):
                _remove_marker(session, backend_id, marker)
                return None
            control["value"] = str(state.get("value", ""))
            control["labels"] = labels

        actionable_nodes.append(
            _ActionableNode(
                ref=ref,
                backend_id=backend_id,
                tag=dom_node.tag,
                attributes=_identity_attributes(
                    dom_node.attributes,
                    exclude_textual=secret,
                ),
                role=role,
                name=public_name,
                operation=operation,
                contexts=contexts,
                in_shadow=dom_node.in_shadow,
                absolute_xpath=dom_node.absolute_xpath,
            )
        )
        control["_backend_id"] = backend_id
        control["_tag"] = dom_node.tag
        control["_role"] = role
        control["_name"] = name
        control["_state"] = _semantic_state(ax_node)
        controls[ref] = control
        return ref, public_name

    def _attach_durable_targets(
        self,
        controls: dict[str, dict[str, object]],
        nodes: tuple[_ActionableNode, ...],
    ) -> None:
        """Attach targets independently; a usable ref may remain non-compilable."""

        for node in nodes:
            try:
                target = self._build_durable_target(
                    node,
                    nodes,
                )
            except (RuntimeError, ValueError):
                target = None
            if target is not None:
                controls[node.ref]["target"] = action_to_step(Click(target))["target"]

    def _build_durable_target(
        self,
        node: _ActionableNode,
        nodes: tuple[_ActionableNode, ...],
    ) -> DurableTarget | None:
        """Build candidates that all confirm the source node through one witness."""

        witness = self._select_witness(node, nodes)
        if witness is None:
            return None
        witness_matches = _matching_nodes(nodes, witness)
        if [match.backend_id for match in witness_matches] != [node.backend_id]:
            return None

        admitted: list[BrowserLocator] = []
        admitted_slots: set[str] = set()
        for locator in _candidate_locators(node, witness):
            slot = _locator_slot(locator)
            if slot in admitted_slots:
                continue
            raw_backend_ids = self._resolve_locator_backend_ids(locator)
            confirmed = {
                match.backend_id
                for match in witness_matches
                if match.backend_id in raw_backend_ids
            }
            if confirmed == {node.backend_id}:
                admitted.append(locator)
                admitted_slots.add(slot)
        if not admitted:
            return None
        return DurableTarget(tuple(admitted), witness)

    def _select_witness(
        self,
        node: _ActionableNode,
        nodes: tuple[_ActionableNode, ...],
    ) -> ElementWitness | None:
        """Select the smallest admitted fact set that uniquely identifies a node."""

        options = _witness_options(node)
        if self._witness_fact_selector is not None:
            try:
                selected = self._witness_fact_selector(
                    tuple(option.public() for option in options)
                )
                if isinstance(selected, str):
                    return None
                selected_ids = tuple(selected)
            except Exception:
                return None
            if not all(isinstance(fact_id, str) for fact_id in selected_ids):
                return None
            if not selected_ids or len(selected_ids) != len(set(selected_ids)):
                return None
            by_id = {option.fact_id: option for option in options}
            if any(fact_id not in by_id for fact_id in selected_ids):
                return None
            witness = _witness_from_options(
                node,
                tuple(by_id[fact_id] for fact_id in selected_ids),
            )
            matches = _matching_nodes(nodes, witness)
            return witness if [item.backend_id for item in matches] == [node.backend_id] else None

        selected: list[_WitnessOption] = []
        witness = _witness_from_options(node, ())
        matches = _matching_nodes(nodes, witness)
        for option in options:
            candidate = _witness_from_options(node, (*selected, option))
            candidate_matches = _matching_nodes(nodes, candidate)
            if selected and len(candidate_matches) >= len(matches):
                continue
            selected.append(option)
            witness = candidate
            matches = candidate_matches
            if [item.backend_id for item in matches] == [node.backend_id]:
                return witness
        return None

    def _resolve_locator_backend_ids(
        self,
        locator: BrowserLocator,
    ) -> tuple[int, ...]:
        """Resolve one locator read-only to backend node identities."""

        if isinstance(locator, RoleLocator):
            return tuple(
                node.backend_id
                for node in self._actionable_nodes
                if node.role == locator.role and node.name == locator.name
            )
        if isinstance(locator, CssLocator):
            page_locator = self._page.locator(locator.selector)
        elif isinstance(locator, XPathLocator):
            page_locator = self._page.locator(f"xpath={locator.expression}")
        else:
            return ()
        return _locator_backend_ids(self._page, page_locator)

    def resolve_durable_target(
        self,
        target: DurableTarget,
        operation: str,
        *,
        label: str | None = None,
    ) -> tuple[_Locator, DurableTarget]:
        """Preflight every candidate on a fresh capture before returning one node."""

        snapshot = self._capture(include_targets=False)
        witness_matches = _matching_nodes(self._actionable_nodes, target.witness)
        if len(witness_matches) != 1:
            raise ValueError("durable target witness is no longer unique")
        intended = witness_matches[0]

        confirmed = []
        confirmed_backend_ids: set[int] = set()
        for locator in target.locators:
            raw_backend_ids = self._resolve_locator_backend_ids(locator)
            matches = {
                node.backend_id
                for node in witness_matches
                if node.backend_id in raw_backend_ids
            }
            if len(matches) > 1:
                raise ValueError("durable locator conflicts with its witness")
            if matches:
                confirmed.append(locator)
                confirmed_backend_ids.update(matches)
        if not confirmed:
            raise ValueError("no durable locator confirms the witnessed target")
        if confirmed_backend_ids != {intended.backend_id}:
            raise ValueError("durable locators resolve to conflicting targets")

        locator = self.resolve_ref(
            snapshot.token,
            intended.ref,
            operation,
            label=label,
        )
        return locator, DurableTarget(tuple(confirmed), target.witness)

    def _remove_previous_markers(self) -> None:
        """Remove the previous snapshot's DOM markers and in-memory identity."""

        if self._marker_attribute is None:
            return
        _remove_page_markers(self._page, self._marker_attribute)
        self._token = None
        self._marker_attribute = None
        self._loader_id = None
        self._bindings = {}
        self._actionable_nodes = ()


def _identity_attributes(
    attributes: Mapping[str, str],
    *,
    exclude_textual: bool = False,
) -> dict[str, str]:
    """Return only admitted identity facts, excluding generated ids."""

    excluded = {"name", "aria-label", "placeholder"} if exclude_textual else set()
    identity = {
        key: value
        for key in _IDENTITY_ATTRIBUTES
        if key not in excluded and (value := attributes.get(key))
    }
    element_id = attributes.get("id")
    if element_id and not looks_generated_id(element_id):
        identity["id"] = element_id
    return identity


def _name_contains_live_value(locator: _Locator, name: str) -> bool:
    """Detect reflected secrets and redact conservatively on inspection errors."""

    if not name:
        return False
    try:
        return bool(
            locator.evaluate(
                """(element, accessibleName) => {
                    const value = String(element.value || '');
                    return Boolean(value) && accessibleName.includes(value);
                }""",
                name,
            )
        )
    except Exception:
        return True


def _witness_options(node: _ActionableNode) -> tuple[_WitnessOption, ...]:
    """List facts in greedy priority order, with ordinary ids deliberately last."""

    options: list[_WitnessOption] = []
    for key in (*_TEST_ATTRIBUTES, "name", "aria-label"):
        if value := node.attributes.get(key):
            options.append(
                _WitnessOption(f"attribute:{key}", "attribute", key, value)
            )
    options.append(
        _WitnessOption(
            "semantic:role-name",
            "semantic",
            "role-name",
            f"{node.role}:{node.name}",
        )
    )
    for key in ("placeholder", "type"):
        if value := node.attributes.get(key):
            options.append(
                _WitnessOption(f"attribute:{key}", "attribute", key, value)
            )
    seen_contexts: set[ContextFact] = set()
    for index, context in enumerate(node.contexts):
        if context.fact in seen_contexts:
            continue
        seen_contexts.add(context.fact)
        options.append(
            _WitnessOption(
                f"context:{index}",
                "context",
                context.fact.relation,
                context.fact.name,
                context.fact,
            )
        )
    if value := node.attributes.get("id"):
        options.append(_WitnessOption("attribute:id", "attribute", "id", value))
    return tuple(options)


def _witness_from_options(
    node: _ActionableNode,
    options: Sequence[_WitnessOption],
) -> ElementWitness:
    """Build a witness from selector-approved observed facts only."""

    attributes: list[tuple[str, str]] = []
    contexts: list[ContextFact] = []
    role: str | None = None
    name: str | None = None
    for option in options:
        if option.kind == "attribute":
            attributes.append((option.key, option.value))
        elif option.kind == "semantic":
            role = node.role
            name = node.name
        elif option.context is not None:
            contexts.append(option.context)
    return ElementWitness(
        tag=node.tag,
        role=role,
        name=name,
        attributes=tuple(attributes),
        context=tuple(contexts),
    )


def _matching_nodes(
    nodes: Sequence[_ActionableNode],
    witness: ElementWitness,
) -> list[_ActionableNode]:
    """Find all actionable nodes that satisfy every witness fact."""

    required_context = set(witness.context)
    return [
        node
        for node in nodes
        if node.tag == witness.tag
        and (witness.role is None or node.role == witness.role)
        and (witness.name is None or node.name == witness.name)
        and all(
            node.attributes.get(key) == value
            for key, value in witness.attributes
        )
        and required_context <= {context.fact for context in node.contexts}
    ]


def _candidate_locators(
    node: _ActionableNode,
    witness: ElementWitness,
) -> tuple[BrowserLocator, ...]:
    """Generate CSS, role, anchored XPath, then absolute XPath candidates."""

    locators: list[BrowserLocator] = []
    for key in (*_TEST_ATTRIBUTES, "id", "name", "aria-label", "placeholder", "type"):
        value = node.attributes.get(key)
        if not value:
            continue
        prefix = "" if key in {*_TEST_ATTRIBUTES, "id"} else node.tag
        try:
            locators.append(
                CssLocator(
                    f"{prefix}[{key}={json.dumps(value, ensure_ascii=False)}]"
                )
            )
        except ValueError:
            continue

    locators.append(RoleLocator(node.role, node.name))
    anchored = _anchored_xpath(node, witness)
    if anchored is not None:
        try:
            locators.append(XPathLocator(anchored, "anchored"))
        except ValueError:
            pass
    if not node.in_shadow and node.absolute_xpath is not None:
        locators.append(XPathLocator(node.absolute_xpath, "absolute"))
    return tuple(locators)


def _locator_slot(locator: BrowserLocator) -> str:
    if isinstance(locator, CssLocator):
        return "css"
    if isinstance(locator, RoleLocator):
        return "role"
    return f"xpath:{locator.mode}"


def _anchored_xpath(
    node: _ActionableNode,
    witness: ElementWitness,
) -> str | None:
    """Anchor a node to one admitted context without fuzzy or class matching."""

    context_by_fact = {context.fact: context for context in node.contexts}
    selected_context = next(
        (
            context_by_fact[fact]
            for fact in witness.context
            if fact in context_by_fact
        ),
        None,
    )
    if selected_context is None:
        return None
    literal = _xpath_literal(selected_context.fact.name)
    if literal is None:
        return None

    leaf = node.tag
    for key, value in witness.attributes:
        leaf_literal = _xpath_literal(value)
        if leaf_literal is not None:
            leaf += f"[@{key}={leaf_literal}]"
            break
    if (
        selected_context.fact.relation == "nearest_heading"
        and re.fullmatch(r"h[1-6]", selected_context.tag)
    ):
        return (
            f"//{selected_context.tag}[normalize-space(.)={literal}]"
            f"/following::{leaf}"
        )
    if selected_context.fact.relation == "ancestor":
        for key in ("aria-label", "name"):
            if selected_context.attributes.get(key) == selected_context.fact.name:
                return (
                    f"//{selected_context.tag}[@{key}={literal}]"
                    f"//{leaf}"
                )
    return None


def _xpath_literal(value: str) -> str | None:
    """Quote a plain XPath literal when one quote style is sufficient."""

    if any(character in value for character in "\r\n\x00"):
        return None
    if '"' not in value:
        return f'"{value}"'
    if "'" not in value:
        return f"'{value}'"
    return None


def _locator_backend_ids(page: _Page, locator: _Locator) -> tuple[int, ...]:
    """Mark locator matches, read their backend ids, and always remove the marker."""

    try:
        count = locator.count()
    except Exception:
        return ()
    if count < 1 or count > _MAX_CONTROLS:
        return ()
    marker = _Marker(
        f"{REF_ATTRIBUTE_PREFIX}candidate-{secrets.token_hex(8)}",
        "match",
    )
    try:
        locator.evaluate_all(
            "(elements, marker) => elements.forEach(element => "
            "element.setAttribute(marker.name, marker.value))",
            {"name": marker.attribute, "value": marker.value},
        )
        session = page.context.new_cdp_session(page)
        try:
            session.send("DOM.enable")
            dom_result = session.send(
                "DOM.getDocument",
                {"depth": -1, "pierce": True},
            )
            root = dom_result.get("root")
            return (
                _backend_ids_with_marker(root, marker)
                if isinstance(root, Mapping)
                else ()
            )
        finally:
            session.detach()
    except Exception:
        return ()
    finally:
        _remove_page_markers(page, marker.attribute)

def _main_frame_identity(session: _CdpSession) -> tuple[str, str]:
    """Read the main frame and loader identities used to detect replacement."""

    result = session.send("Page.getFrameTree")
    frame_tree = result.get("frameTree")
    frame = frame_tree.get("frame") if isinstance(frame_tree, Mapping) else None
    frame_id = frame.get("id") if isinstance(frame, Mapping) else None
    loader_id = frame.get("loaderId") if isinstance(frame, Mapping) else None
    if not isinstance(frame_id, str) or not isinstance(loader_id, str):
        raise RuntimeError("CDP did not provide a main-frame loader identity")
    return frame_id, loader_id


def _dom_revision(page: _Page) -> int:
    """Read the lightweight mutation revision shared with the settle loop."""

    value = page.evaluate(DOM_REVISION_SCRIPT)
    if not isinstance(value, Mapping) or not isinstance(value.get("version"), int):
        raise RuntimeError("browser did not provide a DOM revision")
    return int(value["version"])


def _index_main_document(root: Mapping[str, object]) -> tuple[dict[int, _DomNode], int]:
    """Index main DOM and open shadow roots without descending into iframes."""

    nodes: dict[int, _DomNode] = {}
    closed_roots = 0

    def visit(
        node: Mapping[str, object],
        *,
        shadow_mode: str | None = None,
        in_shadow: bool = False,
        parent_xpath: str | None = None,
        element_index: int = 1,
    ) -> None:
        nonlocal closed_roots
        backend_id = node.get("backendNodeId")
        node_name = node.get("nodeName")
        node_type = node.get("nodeType")
        tag = node_name.lower() if isinstance(node_name, str) else ""
        absolute_xpath = (
            f"{parent_xpath or ''}/{tag}[{element_index}]"
            if node_type == 1 and tag and not in_shadow
            else None
        )
        if isinstance(backend_id, int) and isinstance(node_name, str):
            nodes[backend_id] = _DomNode(
                tag=tag,
                attributes=_dom_attributes(node.get("attributes")),
                closed_shadow=shadow_mode == "closed",
                in_shadow=in_shadow,
                absolute_xpath=absolute_xpath,
            )
        element_counts: dict[str, int] = {}
        for child in _mapping_sequence(node.get("children")):
            child_name = child.get("nodeName")
            child_tag = (
                child_name.lower() if isinstance(child_name, str) else ""
            )
            child_index = 1
            if child.get("nodeType") == 1 and child_tag:
                child_index = element_counts.get(child_tag, 0) + 1
                element_counts[child_tag] = child_index
            visit(
                child,
                shadow_mode=shadow_mode,
                in_shadow=in_shadow,
                parent_xpath=absolute_xpath or parent_xpath,
                element_index=child_index,
            )
        for shadow_root in _mapping_sequence(node.get("shadowRoots")):
            root_mode = str(shadow_root.get("shadowRootType", "")) or shadow_mode
            if root_mode == "closed":
                closed_roots += 1
            visit(
                shadow_root,
                shadow_mode=root_mode,
                in_shadow=True,
            )
        # contentDocument belongs to an iframe and is deliberately unsupported.

    visit(root)
    return nodes, closed_roots


def _dom_attributes(value: object) -> dict[str, str]:
    if not isinstance(value, list):
        return {}
    attributes: dict[str, str] = {}
    for index in range(0, len(value) - 1, 2):
        key = value[index]
        item = value[index + 1]
        if isinstance(key, str) and isinstance(item, str):
            attributes[key] = item
    return attributes


def _backend_ids_with_marker(
    root: Mapping[str, object],
    marker: _Marker,
) -> tuple[int, ...]:
    matches: list[int] = []

    def visit(node: Mapping[str, object]) -> None:
        backend_id = node.get("backendNodeId")
        attributes = _dom_attributes(node.get("attributes"))
        if attributes.get(marker.attribute) == marker.value and isinstance(
            backend_id, int
        ):
            matches.append(backend_id)
        for child in _mapping_sequence(node.get("children")):
            visit(child)
        for shadow_root in _mapping_sequence(node.get("shadowRoots")):
            visit(shadow_root)
        # iframe contentDocument is outside the supported execution surface.

    visit(root)
    return tuple(matches)


def _operation(role: str, tag: str) -> str | None:
    if role == "combobox" and tag == "select":
        return "select_option"
    if role in _FILL_ROLES:
        return "fill"
    if role in _CLICK_ROLES and tag != "option":
        return "click"
    return None


def _set_marker(session: _CdpSession, backend_id: int, marker: _Marker) -> bool:
    try:
        result = session.send("DOM.resolveNode", {"backendNodeId": backend_id})
        remote = result.get("object")
        object_id = remote.get("objectId") if isinstance(remote, Mapping) else None
        if not isinstance(object_id, str):
            return False
        session.send(
            "Runtime.callFunctionOn",
            {
                "objectId": object_id,
                "functionDeclaration": (
                    "function(value) { this.setAttribute('"
                    + marker.attribute
                    + "', value); }"
                ),
                "arguments": [{"value": marker.value}],
            },
        )
        return True
    except Exception:
        return False


def _remove_marker(
    session: _CdpSession,
    backend_id: int,
    marker: _Marker,
) -> None:
    try:
        result = session.send("DOM.resolveNode", {"backendNodeId": backend_id})
        remote = result.get("object")
        object_id = remote.get("objectId") if isinstance(remote, Mapping) else None
        if not isinstance(object_id, str):
            return
        session.send(
            "Runtime.callFunctionOn",
            {
                "objectId": object_id,
                "functionDeclaration": (
                    "function() { this.removeAttribute('"
                    + marker.attribute
                    + "'); }"
                ),
            },
        )
    except Exception:
        pass


def _marker_selector(marker: _Marker) -> str:
    return f"[{marker.attribute}={json.dumps(marker.value)}]"


def _locator_is_live_node(locator: _Locator, marker: _Marker) -> bool:
    try:
        return (
            locator.count() == 1
            and locator.get_attribute(marker.attribute) == marker.value
        )
    except Exception:
        return False


def _remove_page_markers(page: _Page, attribute: str) -> None:
    try:
        page.locator(f"[{attribute}]").evaluate_all(
            "(elements, name) => elements.forEach(element => "
            "element.removeAttribute(name))",
            attribute,
        )
    except Exception:
        pass


def _ax_text(value: object) -> str:
    if not isinstance(value, Mapping):
        return ""
    text = value.get("value")
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _ax_ignored(node: Mapping[str, object]) -> bool:
    return node.get("ignored") is True


def _ax_property(node: Mapping[str, object], name: str) -> object:
    for prop in _mapping_sequence(node.get("properties")):
        if prop.get("name") == name:
            value = prop.get("value")
            return value.get("value") if isinstance(value, Mapping) else None
    return None


def _semantic_state(
    node: Mapping[str, object],
) -> tuple[tuple[str, object], ...]:
    state: list[tuple[str, object]] = []
    for name in ("checked", "selected", "pressed", "expanded", "disabled"):
        value = _ax_property(node, name)
        if value is not None:
            state.append((name, value))
    return tuple(state)


def _backend_semantics(
    session: _CdpSession,
    backend_id: int,
) -> tuple[str, str, tuple[tuple[str, object], ...]] | None:
    """Read current AX role, name, and state for one backend node."""

    result = session.send(
        "Accessibility.getPartialAXTree",
        {"backendNodeId": backend_id, "fetchRelatives": False},
    )
    nodes = result.get("nodes")
    if not isinstance(nodes, list):
        return None
    node = next(
        (
            item
            for item in nodes
            if isinstance(item, Mapping)
            and item.get("backendDOMNodeId") == backend_id
            and not _ax_ignored(item)
        ),
        None,
    )
    if node is None:
        return None
    return (
        _ax_text(node.get("role")).lower(),
        _ax_text(node.get("name")),
        _semantic_state(node),
    )


def _select_labels(locator: _Locator) -> tuple[str, ...]:
    value = locator.evaluate(
        """element => Array.from(element.options)
            .filter(option => !option.disabled && !option.hidden && option.value)
            .map(option => option.textContent.trim())"""
    )
    return tuple(str(label) for label in _sequence(value))


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, (list, tuple)) else ()


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(item for item in _sequence(value) if isinstance(item, Mapping))


def _is_secret(attributes: Mapping[str, str], accessible_name: str) -> bool:
    """Classify controls whose live values must stay out of observations."""

    if attributes.get("type", "").casefold() == "password":
        return True
    autocomplete = attributes.get("autocomplete", "").casefold().split()
    if any(token in _SECRET_AUTOCOMPLETE for token in autocomplete):
        return True
    hints = " ".join(
        (
            accessible_name,
            attributes.get("name", ""),
            attributes.get("aria-label", ""),
            attributes.get("placeholder", ""),
        )
    )
    return _SECRET_TEXT.search(hints) is not None


def _public_dom_state(attributes: Mapping[str, str]) -> dict[str, str]:
    """Expose bounded UI state for decisions, never for durable identity."""

    state: dict[str, str] = {}
    for key in ("aria-pressed", "aria-selected", "aria-checked", "data-state"):
        value = attributes.get(key)
        if value:
            state[key] = value[:100]
    class_name = attributes.get("class", "").strip()
    if class_name:
        state["class"] = class_name[:200]
    return state
