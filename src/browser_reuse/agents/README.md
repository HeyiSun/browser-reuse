# Agents

The formal Agent boundary contains `RecordedAction`, `AgentRun`, and the bounded `run_agent_loop()`. Prompts, response parsers, public-control allowlists, and terminal predicates remain scenario-owned. A caller may opt into treating an explicit model `done` as an unverified success claim; only an outer hard verifier may turn that claim into a verified recipe.
