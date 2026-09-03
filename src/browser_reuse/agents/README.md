# Agents

The formal Agent boundary contains `RecordedAction`, `AgentRun`, and the bounded
`run_agent_loop()`. `ActionPlan` separates the step executed in the current
observation from its optional replay step. `browser.py` adds the site-neutral
DOM+AX prompt, grounded action parser, bounded action-attempt history, and
`run_generic_browser_agent()` proven by two public benchmark sites.

The Agent also retains a bounded list of ref-free semantic facts from earlier
page states. It is deterministic evidence extraction, not model-authored
memory. Revisiting already-seen states twice adds a nudge but never hard-stops
the loop; current controls and refs always come from the fresh observation.

A post-dispatch unresolved action remains in the trajectory for diagnosis but
has no recipe step. The browser Agent may add one mechanically observed exact
`Appears` condition when a named AX fact was absent before the click and unique
after it; this does not use model self-assessment.

For a same-origin link, the compiler may instead attach `UrlIs` only when the
source result exactly matches that link's live href. The href stays hidden from
the model and outside target identity; replay rechecks it before one click.

For a custom combobox popup, `ActionPlan` deliberately executes the temporary
option ref while storing `ChooseComboboxOption(field, exact label)` as the recipe
step. The popup ref never crosses the observation boundary.

A source radio/checkbox gesture also remains `click(ref)`. If its grounded
before/after state proves an exact boolean transition on the same witness, the
post-run compiler replaces only the recipe step with `SetChecked(target, state)`.

An explicit model `done` remains an unverified success claim. An outer hard
verifier may permit candidate compilation; only fresh-context full replay plus
the same verifier makes that candidate verified. Task goals, site bootstrap,
and hard success definitions remain scenario-owned.
