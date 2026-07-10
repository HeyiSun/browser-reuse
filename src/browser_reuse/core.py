"""Small, business-neutral data structures used by browser-reuse."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    goal: str


@dataclass(frozen=True)
class Observation:
    data: Mapping[str, object]


@dataclass(frozen=True)
class Recipe:
    steps: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
