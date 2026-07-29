"""Framework adapters that remain outside browser-reuse core types."""

from .aria import GenericBrowserAdapter
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
from .playwright import BrowserSession, PlaywrightRuntime

__all__ = [
    "action_from_step",
    "action_to_step",
    "BrowserAction",
    "BrowserSession",
    "Click",
    "CssTarget",
    "execute_browser_action",
    "execute_browser_step",
    "Fill",
    "GenericBrowserAdapter",
    "RoleTarget",
    "PlaywrightRuntime",
    "SelectOption",
]
