"""Framework adapters that remain outside browser-reuse core types."""

from .browser import (
    BrowserAction,
    Click,
    CssTarget,
    Fill,
    RoleTarget,
    SelectOption,
    action_from_step,
    action_to_step,
    execute_browser_action,
    execute_browser_step,
)

__all__ = [
    "action_from_step",
    "action_to_step",
    "BrowserAction",
    "Click",
    "CssTarget",
    "execute_browser_action",
    "execute_browser_step",
    "Fill",
    "RoleTarget",
    "SelectOption",
]
