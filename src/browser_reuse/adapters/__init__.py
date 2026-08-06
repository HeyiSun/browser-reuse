"""Framework adapters that remain outside browser-reuse core types."""

from .browser import (
    BrowserAction,
    BrowserLocator,
    Click,
    ContextFact,
    CssLocator,
    DurableTarget,
    ElementWitness,
    Fill,
    RoleLocator,
    SelectOption,
    XPathLocator,
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
    "BrowserLocator",
    "BrowserSession",
    "Click",
    "ContextFact",
    "CssLocator",
    "DurableTarget",
    "ElementWitness",
    "execute_unchecked_browser_action",
    "execute_unchecked_browser_step",
    "Fill",
    "GenericBrowserAdapter",
    "RoleLocator",
    "PlaywrightRuntime",
    "SelectOption",
    "XPathLocator",
]
