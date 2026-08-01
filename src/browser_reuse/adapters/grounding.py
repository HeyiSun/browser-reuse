"""Join Chromium DOM and accessibility trees into executable browser refs."""

from __future__ import annotations

import json
import re
import secrets
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .browser import Click, CssTarget, ElementWitness, RoleTarget, action_to_step


REF_ATTRIBUTE_PREFIX = "data-browser-reuse-ref-"
DOM_REVISION_SCRIPT = f"""() => {{
    const key = '__browser_reuseDomRevisionV1';
    if (!globalThis[key]) {{
        const state = {{version: 0}};
        const observer = new MutationObserver((mutations) => {{
            if (mutations.some(mutation => !(
                mutation.type === 'attributes' &&
                (mutation.attributeName || '').startsWith('{REF_ATTRIBUTE_PREFIX}')
            ))) {{
                state.version += 1;
            }}
        }});
        if (document.documentElement) {{
            observer.observe(document.documentElement, {{
                attributes: true,
                childList: true,
                characterData: true,
                subtree: true,
            }});
        }}
        globalThis[key] = state;
    }}
    return {{ready: document.readyState, version: globalThis[key].version}};
}}"""
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
_TEST_ATTRIBUTES = (
    "data-testid",
    "data-test-id",
    "data-test",
    "data-qa",
    "data-cy",
)
_MAX_SELECT_OPTIONS = 40
_MAX_CONTROLS = 200
_MAX_SNAPSHOT_BYTES = 64_000
_GENERATED_ID_PATTERNS = (
    re.compile(r"^:", re.IGNORECASE),
    re.compile(r":$", re.IGNORECASE),
    re.compile(r"^[0-9a-f]{8,}$", re.IGNORECASE),
    re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
        r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        re.IGNORECASE,
    ),
    re.compile(r"(?:^|[-_])\d{4,}(?:$|[-_])"),
    re.compile(r"^[a-z][a-z0-9_-]*\d{3,}$", re.IGNORECASE),
    re.compile(r"^(?:mantine|radix|headlessui|react)[-_:]", re.IGNORECASE),
)
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

    def evaluate(self, expression: str): ...

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


@dataclass(frozen=True)
class BrowserSnapshot:
    text: str
    controls: Mapping[str, Mapping[str, object]]
    token: str
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class _DomNode:
    tag: str
    attributes: Mapping[str, str]
    closed_shadow: bool


@dataclass(frozen=True)
class _RefBinding:
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
    attribute: str
    value: str


class DomAxGrounder:
    """Capture one main-document DOM+AX view and bind refs to exact live nodes."""

    def __init__(self, page: _Page) -> None:
        self._page = page
        self._token: str | None = None
        self._marker_attribute: str | None = None
        self._loader_id: str | None = None
        self._bindings: dict[str, _RefBinding] = {}

    def capture(self) -> BrowserSnapshot:
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
                raise RuntimeError("document changed while capturing DOM and AX")

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
                    depth=0,
                    visited=visited,
                )

            _, loader_final = _main_frame_identity(session)
            if loader_before != loader_final:
                raise RuntimeError("document changed while publishing browser refs")
            if _dom_revision(self._page) != revision_before:
                raise RuntimeError("document mutated while capturing DOM and AX")
        except Exception:
            _remove_page_markers(self._page, marker_attribute)
            raise
        finally:
            session.detach()

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
        if token != self._token or self._loader_id is None:
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

    def semantic_witness_matches(
        self,
        locator: _Locator,
        witness: ElementWitness,
    ) -> bool:
        if witness.role is None or witness.name is None:
            return True
        attribute = f"{REF_ATTRIBUTE_PREFIX}verify-{secrets.token_hex(8)}"
        marker = _Marker(attribute, "target")
        try:
            locator.evaluate(
                "(element, marker) => element.setAttribute(marker.name, marker.value)",
                {"name": marker.attribute, "value": marker.value},
            )
            session = self._page.context.new_cdp_session(self._page)
            try:
                session.send("DOM.enable")
                session.send("Accessibility.enable")
                dom_result = session.send(
                    "DOM.getDocument",
                    {"depth": -1, "pierce": True},
                )
                dom_root = dom_result.get("root")
                backend_ids = (
                    _backend_ids_with_marker(dom_root, marker)
                    if isinstance(dom_root, Mapping)
                    else ()
                )
                if len(backend_ids) != 1:
                    return False
                semantics = _backend_semantics(session, backend_ids[0])
            finally:
                session.detach()
            return semantics[0:2] == (witness.role, witness.name)
        except Exception:
            return False
        finally:
            try:
                locator.evaluate(
                    "(element, name) => element.removeAttribute(name)",
                    marker.attribute,
                )
            except Exception:
                pass

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
        *,
        depth: int,
        visited: set[str],
    ) -> None:
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

        ref: str | None = None
        if dom_node is not None and not _ax_ignored(node):
            ref = self._publish_control(
                node,
                dom_node,
                int(backend_id),
                session,
                marker_attribute,
                controls,
            )

        child_ids = [str(value) for value in _sequence(node.get("childIds"))]
        if dom_node is not None and _is_secret(dom_node.attributes, name):
            child_ids = []
        show = bool(role and (name or ref or role not in _STRUCTURAL_ROLES))
        child_depth = depth + 1 if show else depth
        if show:
            line = "  " * depth + f"- {role}"
            if name:
                line += f" {json.dumps(name, ensure_ascii=False)}"
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
    ) -> str | None:
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

        control: dict[str, object] = {"op": operation, "name": name}
        if operation == "fill" and not _is_secret(dom_node.attributes, name):
            control["value"] = locator.input_value()
        elif operation == "select_option":
            if _is_secret(dom_node.attributes, name):
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

        target = self._durable_target(
            marker,
            dom_node,
            role,
            name,
        )
        if target is not None:
            control["target"] = action_to_step(Click(target))["target"]
        control["_backend_id"] = backend_id
        control["_tag"] = dom_node.tag
        control["_role"] = role
        control["_name"] = name
        control["_state"] = _semantic_state(ax_node)
        controls[ref] = control
        return ref

    def _durable_target(
        self,
        marker: _Marker,
        dom_node: _DomNode,
        role: str,
        name: str,
    ) -> CssTarget | RoleTarget | None:
        attributes = dom_node.attributes

        for attribute in _TEST_ATTRIBUTES:
            value = attributes.get(attribute)
            target = self._attribute_target(
                dom_node,
                role,
                name,
                marker,
                attribute,
                value,
                include_tag=False,
            )
            if target is not None:
                return target

        element_id = attributes.get("id")
        if element_id and not looks_generated_id(element_id):
            target = self._attribute_target(
                dom_node,
                role,
                name,
                marker,
                "id",
                element_id,
                include_tag=False,
            )
            if target is not None:
                return target

        if name:
            locator = self._page.get_by_role(role, name=name, exact=True)
            if _locator_is_live_node(locator, marker):
                return RoleTarget(role, name)

        for attribute in ("name", "aria-label", "placeholder"):
            target = self._attribute_target(
                dom_node,
                role,
                name,
                marker,
                attribute,
                attributes.get(attribute),
                include_tag=True,
            )
            if target is not None:
                return target
        return None

    def _attribute_target(
        self,
        dom_node: _DomNode,
        role: str,
        name: str,
        marker: _Marker,
        attribute: str,
        value: str | None,
        *,
        include_tag: bool,
    ) -> CssTarget | None:
        if not value:
            return None
        prefix = dom_node.tag if include_tag else ""
        selector = f"{prefix}[{attribute}={json.dumps(value, ensure_ascii=False)}]"
        try:
            locator = self._page.locator(selector)
            if not _locator_is_live_node(locator, marker):
                return None
            witness = ElementWitness(
                tag=dom_node.tag,
                role=role if name else None,
                name=name or None,
                attributes=((attribute, value),),
            )
            return CssTarget(selector, witness)
        except ValueError:
            return None

    def _remove_previous_markers(self) -> None:
        if self._marker_attribute is None:
            return
        _remove_page_markers(self._page, self._marker_attribute)
        self._token = None
        self._marker_attribute = None
        self._loader_id = None
        self._bindings = {}


def looks_generated_id(value: str) -> bool:
    """Return whether an HTML id looks allocated by a UI runtime."""

    return any(pattern.search(value) for pattern in _GENERATED_ID_PATTERNS)


def witness_matches(locator: _Locator, witness: ElementWitness) -> bool:
    """Check the minimal recorded identity before a replay side effect."""

    if locator.count() != 1:
        return False
    value = locator.evaluate(
        """element => ({
            tag: element.tagName.toLowerCase(),
            attributes: Object.fromEntries(Array.from(element.attributes)
                .map(attribute => [attribute.name, attribute.value]))
        })"""
    )
    if not isinstance(value, Mapping) or value.get("tag") != witness.tag:
        return False
    attributes = value.get("attributes")
    if not isinstance(attributes, Mapping):
        return False
    return all(attributes.get(key) == expected for key, expected in witness.attributes)


def _main_frame_identity(session: _CdpSession) -> tuple[str, str]:
    result = session.send("Page.getFrameTree")
    frame_tree = result.get("frameTree")
    frame = frame_tree.get("frame") if isinstance(frame_tree, Mapping) else None
    frame_id = frame.get("id") if isinstance(frame, Mapping) else None
    loader_id = frame.get("loaderId") if isinstance(frame, Mapping) else None
    if not isinstance(frame_id, str) or not isinstance(loader_id, str):
        raise RuntimeError("CDP did not provide a main-frame loader identity")
    return frame_id, loader_id


def _dom_revision(page: _Page) -> int:
    value = page.evaluate(DOM_REVISION_SCRIPT)
    if not isinstance(value, Mapping) or not isinstance(value.get("version"), int):
        raise RuntimeError("browser did not provide a DOM revision")
    return int(value["version"])


def _index_main_document(root: Mapping[str, object]) -> tuple[dict[int, _DomNode], int]:
    nodes: dict[int, _DomNode] = {}
    closed_roots = 0

    def visit(node: Mapping[str, object], *, shadow_mode: str | None = None) -> None:
        nonlocal closed_roots
        backend_id = node.get("backendNodeId")
        node_name = node.get("nodeName")
        if isinstance(backend_id, int) and isinstance(node_name, str):
            nodes[backend_id] = _DomNode(
                tag=node_name.lower(),
                attributes=_dom_attributes(node.get("attributes")),
                closed_shadow=shadow_mode == "closed",
            )
        for child in _mapping_sequence(node.get("children")):
            visit(child, shadow_mode=shadow_mode)
        for shadow_root in _mapping_sequence(node.get("shadowRoots")):
            root_mode = str(shadow_root.get("shadowRootType", "")) or shadow_mode
            if root_mode == "closed":
                closed_roots += 1
            visit(shadow_root, shadow_mode=root_mode)
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
