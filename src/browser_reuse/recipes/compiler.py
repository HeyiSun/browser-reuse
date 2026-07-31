"""Compile an observed successful action sequence into a finite recipe."""

from collections.abc import Sequence

from browser_reuse.agents import RecordedAction
from browser_reuse.agents.loop import AgentRun
from browser_reuse.core import ExecutionResult, Recipe

from .serialization import snapshot_step


def compile_recipe(actions: Sequence[RecordedAction]) -> Recipe:
    """Deep-snapshot every replayable recipe step in order."""

    recipe_steps: list[dict[str, object]] = []
    for action in actions:
        if action.recipe_step is None:
            raise ValueError("trajectory contains an action without a recipe step")
        recipe_steps.append(snapshot_step(action.recipe_step))

    return Recipe(steps=tuple(recipe_steps))


def compile_verified_recipe(
    run: AgentRun,
    result: ExecutionResult,
) -> Recipe | None:
    if not run.claimed_success or run.error is not None or not result.success:
        return None
    if any(action.recipe_step is None for action in run.actions):
        return None
    return compile_recipe(run.actions)
