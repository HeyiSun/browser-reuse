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


REF_ATTRIBUTE = "data-browser_reuse-ref"
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
    loader_id: str
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class _DomNode:
    tag: str
    attributes: Mapping[str, str]
    closed_shadow: bool


@dataclass(frozen=True)
class _RefBinding:
    ref: str
    marker: str
    operation: str
    backend_id: int


class DomAxGrounder:
    """Capture one main-document DOM+AX view and bind refs to exact live nodes."""

    def __init__(self, page: _Page) -> None:
        self._page = page
        self._token: str | None = None
        self._loader_id: str | None = None
        self._bindings: dict[str, _RefBinding] = {}

    def capture(self) -> BrowserSnapshot:
        started = time.perf_counter()
        self._remove_previous_markers()
        token = secrets.token_hex(8)
        session = self._page.context.new_cdp_session(self._page)
        try:
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
                    token,
                    controls,
                    diagnostics,
                    lines,
                    depth=0,
                    visited=visited,
                )

            _, loader_final = _main_frame_identity(session)
            if loader_before != loader_final:
                raise RuntimeError("document changed while publishing browser refs")
        finally:
            session.detach()

        self._token = token
        self._loader_id = loader_before
        bindings: dict[str, _RefBinding] = {}
        for ref, control in controls.items():
            backend_id = control.pop("_backend_id", None)
            if not isinstance(backend_id, int):
                raise RuntimeError("published browser ref lost its backend identity")
            bindings[ref] = _RefBinding(
                ref=ref,
                marker=f"{token}:{ref}",
                operation=str(control["op"]),
                backend_id=backend_id,
            )
        self._bindings = bindings
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
        return BrowserSnapshot(
            text=text,
            controls=controls,
            token=token,
            loader_id=loader_before,
            diagnostics=diagnostics,
        )

    def resolve_ref(self, token: str, ref: str, operation: str) -> _Locator:
        if token != self._token or self._loader_id is None:
            raise ValueError("browser ref belongs to a stale observation")
        binding = self._bindings.get(ref)
        if binding is None or binding.operation != operation:
            raise ValueError("browser ref/action is not present in the observation")

        session = self._page.context.new_cdp_session(self._page)
        try:
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
        finally:
            session.detach()
        if loader_id != self._loader_id:
            raise ValueError("browser ref belongs to a replaced document")
        if live_backend_ids != (binding.backend_id,):
            raise ValueError("browser ref no longer resolves to its live node")

        locator = self._page.locator(_marker_selector(binding.marker))
        if locator.count() != 1:
            raise ValueError("browser ref no longer resolves to its live node")
        if not locator.is_visible() or not locator.is_enabled():
            raise ValueError("browser ref is no longer actionable")
        if operation == "fill" and not locator.is_editable():
            raise ValueError("browser ref is no longer editable")
        return locator

    def _render_ax_node(
        self,
        node_id: str,
        ax_by_id: Mapping[str, Mapping[str, object]],
        dom_nodes: Mapping[int, _DomNode],
        session: _CdpSession,
        token: str,
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
                token,
                controls,
            )

        child_ids = [str(value) for value in _sequence(node.get("childIds"))]
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
                token,
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
        token: str,
        controls: dict[str, dict[str, object]],
    ) -> str | None:
        role = _ax_text(ax_node.get("role")).lower()
        name = _ax_text(ax_node.get("name"))
        operation = _operation(role, dom_node.tag)
        if operation is None or dom_node.closed_shadow:
            return None

        ref = f"e{len(controls)}"
        marker = f"{token}:{ref}"
        if not _set_marker(session, backend_id, marker):
            return None
        locator = self._page.locator(_marker_selector(marker))
        if (
            locator.count() != 1
            or not locator.is_visible()
            or not locator.is_enabled()
            or (operation == "fill" and not locator.is_editable())
        ):
            _remove_marker(session, backend_id)
            return None

        control: dict[str, object] = {"op": operation, "name": name}
        if operation == "fill" and not _is_secret(dom_node.attributes, name):
            control["value"] = locator.input_value()
        elif operation == "select_option":
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
                _remove_marker(session, backend_id)
                return None
            labels = tuple(str(label) for label in _sequence(state.get("labels")))
            if (
                not labels
                or len(labels) > _MAX_SELECT_OPTIONS
                or len(labels) != len(set(labels))
            ):
                _remove_marker(session, backend_id)
                return None
            control["value"] = str(state.get("value", ""))
            control["labels"] = labels

        target = self._durable_target(
            locator,
            marker,
            dom_node,
            role,
            name,
        )
        if target is not None:
            control["target"] = action_to_step(Click(target))["target"]
        control["_backend_id"] = backend_id
        controls[ref] = control
        return ref

    def _durable_target(
        self,
        live_locator: _Locator,
        marker: str,
        dom_node: _DomNode,
        role: str,
        name: str,
    ) -> CssTarget | RoleTarget | None:
        del live_locator
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
        marker: str,
        attribute: str,
        value: str | None,
        *,
        include_tag: bool,
    ) -> CssTarget | None:
        if not value:
            return None
        prefix = dom_node.tag if include_tag else ""
        selector = f"{prefix}[{attribute}={json.dumps(value)}]"
        locator = self._page.locator(selector)
        if not _locator_is_live_node(locator, marker):
            return None
        witness = ElementWitness(
            tag=dom_node.tag,
            role=role or None,
            name=name or None,
            attributes=((attribute, value),),
        )
        return CssTarget(selector, witness)

    def _remove_previous_markers(self) -> None:
        if self._token is None:
            return
        try:
            self._page.locator(f"[{REF_ATTRIBUTE}]").evaluate_all(
                """(elements, prefix) => {
                    for (const element of elements) {
                        if ((element.getAttribute('data-browser_reuse-ref') || '')
                            .startsWith(prefix)) {
                            element.removeAttribute('data-browser_reuse-ref');
                        }
                    }
                }""",
                f"{self._token}:",
            )
        except Exception:
            pass
        self._token = None
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
    marker: str,
) -> tuple[int, ...]:
    matches: list[int] = []

    def visit(node: Mapping[str, object]) -> None:
        backend_id = node.get("backendNodeId")
        attributes = _dom_attributes(node.get("attributes"))
        if attributes.get(REF_ATTRIBUTE) == marker and isinstance(backend_id, int):
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


def _set_marker(session: _CdpSession, backend_id: int, marker: str) -> bool:
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
                    + REF_ATTRIBUTE
                    + "', value); }"
                ),
                "arguments": [{"value": marker}],
            },
        )
        return True
    except Exception:
        return False


def _remove_marker(session: _CdpSession, backend_id: int) -> None:
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
                    + REF_ATTRIBUTE
                    + "'); }"
                ),
            },
        )
    except Exception:
        pass


def _marker_selector(marker: str) -> str:
    return f"[{REF_ATTRIBUTE}={json.dumps(marker)}]"


def _locator_is_live_node(locator: _Locator, marker: str) -> bool:
    try:
        return locator.count() == 1 and locator.get_attribute(REF_ATTRIBUTE) == marker
    except Exception:
        return False


def _ax_text(value: object) -> str:
    if not isinstance(value, Mapping):
        return ""
    text = value.get("value")
    return str(text) if text is not None else ""


def _ax_ignored(node: Mapping[str, object]) -> bool:
    return node.get("ignored") is True


def _ax_property(node: Mapping[str, object], name: str) -> object:
    for prop in _mapping_sequence(node.get("properties")):
        if prop.get("name") == name:
            value = prop.get("value")
            return value.get("value") if isinstance(value, Mapping) else None
    return None


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
