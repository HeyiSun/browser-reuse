# Browser subsystem

This package contains the site-neutral browser integration. Each filename names
one responsibility:

| Module | Responsibility |
|---|---|
| `actions.py` | `Click`, `Fill`, `SelectOption`, and the strict step codec |
| `targets.py` | Locator candidates, witnesses, durable targets, and validation |
| `dom_ax.py` | DOM+AX capture, snapshot refs, target building, and replay preflight |
| `adapter.py` | `DomAxBrowserAdapter.observe()` and `execute()` |
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

The current grounder supports the main document and open shadow DOM. Iframes,
OOPIFs, and closed shadow DOM fail closed. Navigation bootstrap, business tasks,
and verifiers remain outside this package.
