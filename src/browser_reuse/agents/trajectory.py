"""Framework-neutral plan and record for one Agent action."""

from collections.abc import Mapping
from dataclasses import dataclass

from browser_reuse.core import Observation


@dataclass(frozen=True)
class ActionPlan:
    """One immediate adapter step and its optional replayable counterpart."""

    execute_step: Mapping[str, object]
    recipe_step: Mapping[str, object] | None


@dataclass(frozen=True)
class RecordedAction:
    """A completed plan together with its public before/after observations."""

    execute_step: Mapping[str, object]
    recipe_step: Mapping[str, object] | None
    before: Observation
    after: Observation | None
