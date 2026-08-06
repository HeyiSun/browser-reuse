"""Framework-neutral interfaces for observing, acting, and verifying."""

from collections.abc import Mapping
from typing import Protocol

from .core import Observation, TaskSpec


class ActionDispatchedError(RuntimeError):
    """The side effect returned, but post-action settling failed."""


class Adapter(Protocol):
    def observe(self) -> Observation:
        ...

    def execute(
        self,
        step: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        """Execute one step and optionally return the canonical effective step."""

        ...


class Verifier(Protocol):
    def verify(self, task: TaskSpec) -> bool:
        ...
