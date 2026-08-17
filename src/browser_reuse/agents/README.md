# Agents

The formal Agent boundary contains `RecordedAction`, `AgentRun`, and the bounded
`run_agent_loop()`. `ActionPlan` separates the step executed in the current
observation from its optional replay step. `browser.py` adds the site-neutral
DOM+AX prompt, grounded action parser, bounded action-attempt history, and
`run_generic_browser_agent()` proven by two public benchmark sites.

A post-dispatch unresolved action remains in the trajectory for diagnosis but
has no recipe step. The browser Agent may add one mechanically observed exact
`Appears` condition when a named AX fact was absent before the click and unique
after it; this does not use model self-assessment.

An explicit model `done` remains an unverified success claim. An outer hard
verifier may permit candidate compilation; only fresh-context full replay plus
the same verifier makes that candidate verified. Task goals, site bootstrap,
and hard success definitions remain scenario-owned.
