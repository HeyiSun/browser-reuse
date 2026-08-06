"""Execute a recipe without making new Agent decisions."""

from collections.abc import Mapping
from dataclasses import dataclass

from browser_reuse.core import Recipe
from browser_reuse.interfaces import Adapter

from .serialization import snapshot_step


@dataclass(frozen=True)
class ReplayProgress:
    """Steps whose execute call returned, plus the first execution error."""

    completed_steps: int
    effective_steps: tuple[Mapping[str, object], ...]
    error: Exception | None


def try_replay_recipe(recipe: Recipe, adapter: Adapter) -> ReplayProgress:
    """Execute until the first error and expose the successful prefix length."""

    effective_steps: list[Mapping[str, object]] = []
    for index, step in enumerate(recipe.steps):
        try:
            effective_step = adapter.execute(step)
            if effective_step is not None and not isinstance(
                effective_step,
                Mapping,
            ):
                raise ValueError("adapter returned a non-mapping replay receipt")
            canonical = snapshot_step(
                effective_step if effective_step is not None else step
            )
        except Exception as exc:
            return ReplayProgress(
                completed_steps=index,
                effective_steps=tuple(effective_steps),
                error=exc,
            )
        effective_steps.append(canonical)
    return ReplayProgress(
        completed_steps=len(recipe.steps),
        effective_steps=tuple(effective_steps),
        error=None,
    )


def replay_recipe(recipe: Recipe, adapter: Adapter) -> None:
    """Execute steps in order and let the first adapter error stop replay."""

    progress = try_replay_recipe(recipe, adapter)
    if progress.error is not None:
        raise progress.error
