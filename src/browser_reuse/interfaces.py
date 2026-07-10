"""Framework-neutral interfaces for observing, acting, and verifying."""

from collections.abc import Mapping
from typing import Protocol

from .core import Observation, TaskSpec


class Adapter(Protocol):
    def observe(self) -> Observation:
        ...

    def execute(self, step: Mapping[str, object]) -> None:
        ...


class Verifier(Protocol):
    def verify(self, task: TaskSpec) -> bool:
        ...
