# Browser subsystem

This package contains the site-neutral browser integration. Each filename names
one responsibility:

| Module | Responsibility |
|---|---|
| `actions.py` | Typed actions, exact `Appears` readback, and the strict codec |
| `targets.py` | Locator candidates, witnesses, durable targets, and validation |
| `dom_ax.py` | DOM+AX capture, refs, targets, AX facts, and replay preflight |
| `adapter.py` | Browser execution plus field-owned and `Appears` readback |
| `runtime.py` | Playwright and fresh browser-context ownership |

The dependency direction stays simple:

```text
targets ← actions
   ↑         ↑
 dom_ax ─────┘
   ↑
 adapter        runtime
```

`browser.__init__` exposes only the adapter and runtime. Callers that construct
actions or targets import their defining module, so dependencies remain visible.

Snapshot refs and durable recipe targets are separate. A source action may be
executable without being compilable; fresh-context replay and the hard verifier
decide whether a recipe can be published.

Fill and option actions confirm their own field state. A click may optionally
require one exact bounded named AX outcome fact that was absent before source
dispatch and appeared afterward. This includes semantic page states and named
interactive facts, but excludes long composite names. Page quiet and URL change
are not readback. Runtime values remain outcome evidence and never become target
identity.

For custom comboboxes, a recipe stores only the witnessed field and exact option
text. Editable fields are filled; readonly fields are opened. The popup option is
freshly observed, gets only a snapshot-local ref, and is clicked once. Replay then
confirms field-owned selected/value evidence. Ambiguous owners or options fail
before dispatch.

The Agent still uses a plain click for a checkbox or radio. When source
before/after observations prove a boolean transition on the same witnessed
control, the compiler stores `SetChecked` instead. Replay treats it as a target
state: an already-satisfied target is a zero-click success; otherwise the exact
target is clicked once and reread locally, including after a DOM remount. It
never retries the click or waits for unrelated page quiet.

The current grounder supports the main document and open shadow DOM. Iframes,
OOPIFs, and closed shadow DOM fail closed. Navigation bootstrap, business tasks,
and verifiers remain outside this package.
