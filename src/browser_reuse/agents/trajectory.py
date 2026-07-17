"""Framework-neutral record of one successfully executed Agent action."""

from collections.abc import Mapping
from dataclasses import dataclass

from browser_reuse.core import Observation


@dataclass(frozen=True)
class RecordedAction:
    """An adapter action together with its public before/after observations."""

    action: Mapping[str, object]
    before: Observation
    after: Observation
