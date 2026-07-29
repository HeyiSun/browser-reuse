# Agents

The formal Agent boundary contains `RecordedAction`, `AgentRun`, and the bounded
`run_agent_loop()`. `browser.py` adds the site-neutral public-ARIA prompt,
grounded action parser, bounded successful-action history, and
`run_generic_browser_agent()` proven by two public benchmark sites.

An explicit model `done` remains an unverified success claim; only an outer hard
verifier may turn that claim into a verified recipe. Task goals, site bootstrap,
and hard success definitions remain scenario-owned.
