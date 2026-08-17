"""Framework-neutral interfaces for observing, acting, and verifying."""

from collections.abc import Mapping
from typing import Protocol

from .core import Observation, TaskSpec


class ActionDispatchedError(RuntimeError):
    """The action was dispatched, but its expected effect is unresolved."""


class ActionNotCommittedError(RuntimeError):
    """A recoverable semantic action stopped before its commit point."""


class Adapter(Protocol):
    def observe(self) -> Observation:
        ...

    def execute(
        self,
        step: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        """Execute and confirm one step, optionally returning its effective form."""

        ...


class Verifier(Protocol):
    def verify(self, task: TaskSpec) -> bool:
        ...
