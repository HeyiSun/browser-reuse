# Recipes

The formal recipe boundary contains verified compilation, canonical
JSON-compatible payload conversion, deterministic replay, and fail-closed
replay progress. Compilation preserves every successful replay step and refuses
an otherwise successful source trajectory when any executed action has no
durable counterpart. Snapshot refs never enter recipes. General locator
fallback and Hybrid recovery remain scenario-owned.
