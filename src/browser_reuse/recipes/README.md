# Recipes

Phase 2 currently exposes `compile_recipe()` and `replay_recipe()`. Compilation preserves state-changing recorded actions and removes exact observed no-ops; replay executes the finite steps in order and stops on the first adapter error. General locator fallback and recovery remain later work.
