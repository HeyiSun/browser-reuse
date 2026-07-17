"""Execute a recipe without making new Agent decisions."""

from browser_reuse.core import Recipe
from browser_reuse.interfaces import Adapter


def replay_recipe(recipe: Recipe, adapter: Adapter) -> None:
    """Execute steps in order and let the first adapter error stop replay."""

    for step in recipe.steps:
        adapter.execute(step)
