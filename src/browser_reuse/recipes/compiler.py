"""Compile an observed successful action sequence into a finite recipe."""

from collections.abc import Sequence

from browser_reuse.agents import RecordedAction
from browser_reuse.agents.loop import AgentRun
from browser_reuse.core import ExecutionResult, Recipe

from .serialization import snapshot_step


def compile_recipe(actions: Sequence[RecordedAction]) -> Recipe:
    """Deep-snapshot every successfully executed action in order."""

    return Recipe(
        steps=tuple(snapshot_step(record.action) for record in actions)
    )


def compile_verified_recipe(
    run: AgentRun,
    result: ExecutionResult,
) -> Recipe | None:
    if not run.claimed_success or run.error is not None or not result.success:
        return None
    return compile_recipe(run.actions)
