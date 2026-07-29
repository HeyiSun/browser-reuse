"""Small framework-neutral Agent loop and trajectory records."""

from .browser import GENERIC_BROWSER_PROMPT, run_generic_browser_agent
from .loop import AgentRun, run_agent_loop
from .trajectory import RecordedAction

__all__ = [
    "AgentRun",
    "GENERIC_BROWSER_PROMPT",
    "RecordedAction",
    "run_agent_loop",
    "run_generic_browser_agent",
]
