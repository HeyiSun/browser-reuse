# Recipes

The recipe boundary contains source-verified candidate compilation, canonical
JSON-compatible payload conversion, deterministic replay, and replay progress.
Compilation preserves every successful replay step and refuses a source
trajectory when any executed action has no durable counterpart. Snapshot refs
never enter recipes.

A candidate is verified only after full replay in a fresh context and the same
independent task verifier both pass. Replay stops on its first error; it never
invokes an Agent. The browser adapter validates stored locator candidates against
the same mechanical witness before dispatch. Optional model-proposed locator
hints require explicit adapter configuration and are disabled by default.
