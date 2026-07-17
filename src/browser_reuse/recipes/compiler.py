"""Compile an observed successful action sequence into a finite recipe."""

from collections.abc import Sequence

from browser_reuse.agents import RecordedAction
from browser_reuse.core import Recipe


def compile_recipe(actions: Sequence[RecordedAction]) -> Recipe:
    """Preserve state-changing actions and remove exact observed no-ops."""

    return Recipe(
        steps=tuple(
            dict(record.action)
            for record in actions
            if record.before != record.after
        )
    )
