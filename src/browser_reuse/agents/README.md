# Agents

The formal Agent boundary contains `RecordedAction`, `AgentRun`, and the bounded
`run_agent_loop()`. `ActionPlan` separates the step executed in the current
observation from its optional replay step. `browser.py` adds the site-neutral
DOM+AX prompt, grounded action parser, bounded successful-action history, and
`run_generic_browser_agent()` proven by two public benchmark sites.

An explicit model `done` remains an unverified success claim. An outer hard
verifier may permit candidate compilation; only fresh-context full replay plus
the same verifier makes that candidate verified. Task goals, site bootstrap,
and hard success definitions remain scenario-owned.
