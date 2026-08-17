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
require one exact named AX heading, alert, status, or dialog that was absent
before source dispatch and appeared afterward. Page quiet and URL change are not
readback. Runtime values remain outcome evidence and never become target identity.

The current grounder supports the main document and open shadow DOM. Iframes,
OOPIFs, and closed shadow DOM fail closed. Navigation bootstrap, business tasks,
and verifiers remain outside this package.
