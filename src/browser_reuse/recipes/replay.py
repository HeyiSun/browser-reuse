"""Execute a recipe without making new Agent decisions."""

from dataclasses import dataclass

from browser_reuse.core import Recipe
from browser_reuse.interfaces import Adapter


@dataclass(frozen=True)
class ReplayProgress:
    """Steps whose execute call returned, plus the first execution error."""

    completed_steps: int
    error: Exception | None


def try_replay_recipe(recipe: Recipe, adapter: Adapter) -> ReplayProgress:
    """Execute until the first error and expose the successful prefix length."""

    for index, step in enumerate(recipe.steps):
        try:
            adapter.execute(step)
        except Exception as exc:
            return ReplayProgress(completed_steps=index, error=exc)
    return ReplayProgress(completed_steps=len(recipe.steps), error=None)


def replay_recipe(recipe: Recipe, adapter: Adapter) -> None:
    """Execute steps in order and let the first adapter error stop replay."""

    progress = try_replay_recipe(recipe, adapter)
    if progress.error is not None:
        raise progress.error
