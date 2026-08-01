"""Framework adapters that remain outside browser-reuse core types."""

from .browser import (
    BrowserAction,
    Click,
    CssTarget,
    ElementWitness,
    Fill,
    RoleTarget,
    SelectOption,
    action_from_step,
    action_to_step,
    execute_unchecked_browser_action,
    execute_unchecked_browser_step,
)
from .generic import GenericBrowserAdapter
from .playwright import BrowserSession, PlaywrightRuntime

__all__ = [
    "action_from_step",
    "action_to_step",
    "BrowserAction",
    "BrowserSession",
    "Click",
    "CssTarget",
    "ElementWitness",
    "execute_unchecked_browser_action",
    "execute_unchecked_browser_step",
    "Fill",
    "GenericBrowserAdapter",
    "RoleTarget",
    "PlaywrightRuntime",
    "SelectOption",
]
