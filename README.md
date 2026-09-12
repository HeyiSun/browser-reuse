# browser-reuse

A small Python research runtime for exploring browser tasks with an Agent and
turning successful trajectories into reusable recipes.

```text
Task + browser observation → Agent actions → recorded trajectory
    → independent task verification → candidate recipe
    → fresh-context replay + task verification → verified recipe
```

The runtime separates page observation, action execution, recipe compilation,
and task verification. An Agent's `done` response is a success claim; the caller's
verifier determines whether the task actually succeeded.

## Install

Requires Python 3.10 or newer. From this repository:

```bash
python -m pip install -e '.[browser,llm]'
python -m playwright install chromium
```

The base package has no runtime dependencies. The `browser` extra adds Playwright;
the `llm` extra adds LiteLLM. Install only the extras your application uses.

## Observe a page

This example uses an in-memory page and makes no model call:

```python
from browser_reuse.browser import DomAxBrowserAdapter, PlaywrightRuntime

with PlaywrightRuntime() as runtime:
    with runtime.new_session() as session:
        session.page.set_content('<button aria-label="Continue">Continue</button>')
        adapter = DomAxBrowserAdapter(session.page)
        observation = adapter.observe()
        print(observation.data)
```

## Configure a model

`load_chat_model()` accepts `provider:model` and delegates requests to LiteLLM.
Set `BROWSER_REUSE_MODEL` to the provider and model available in your account,
and configure credentials through that provider's standard environment variables.
For example, an OpenAI provider uses `OPENAI_API_KEY`.

```python
from browser_reuse import Message, load_chat_model

model = load_chat_model()  # Reads BROWSER_REUSE_MODEL.
reply = model.complete([Message(role="user", content="Hello")])
```

An explicit `load_chat_model("provider:model-name")` overrides the environment.
The model name above is a placeholder. No model or credential is bundled.
The package does not load `.env` files; supply configuration through your shell
or your application's configuration layer. Keep credentials out of source,
trajectory files, and Git.

## Runtime API

| Module | Responsibility |
|---|---|
| `browser_reuse.core` | `TaskSpec`, `Observation`, `Recipe`, `ExecutionResult` |
| `browser_reuse.interfaces` | Adapter/verifier contracts and action error semantics |
| `browser_reuse.llm` | Provider-neutral `ChatModel`, `Message`, and model loading |
| `browser_reuse.agents` | Bounded Agent loop and action trajectory recording |
| `browser_reuse.browser` | DOM + accessibility grounding and Playwright lifecycle |
| `browser_reuse.browser.actions` | Click, fill, select, combobox, checked-state, and readback contracts |
| `browser_reuse.browser.targets` | Durable locator candidates and mechanical witnesses |
| `browser_reuse.recipes` | Candidate compilation, serialization, and ordered replay |

Run an exploration with `run_generic_browser_agent(task, adapter, model)`.
After an independent verifier checks the source result, pass the `AgentRun` and
`ExecutionResult` to `compile_candidate_recipe()`. A returned recipe is still a
candidate: reset the task environment, create a fresh browser context, replay it
with `try_replay_recipe()`, and run the same verifier before treating it as verified.
Environment reset, navigation, persistence, and task-specific verifiers belong
to the caller.

## Current boundaries

- Chromium main document and open shadow DOM are supported. Iframes, OOPIFs, and
  closed shadow DOM are unsupported and fail closed.
- Snapshot-local refs identify current elements; durable recipes store locator
  candidates plus a mechanical witness. Ambiguity stops execution.
- Fill and option actions use field-owned readback. Fill, native select, and
  recipe-only `SetChecked` avoid dispatch when the desired state already holds.
  A click can require an exact bounded accessibility fact to appear, or a
  same-origin link's exact destination to be reached. Page quiet alone does not
  prove success.
- A dispatched but unresolved action is recorded and cannot become a recipe step.
  Replay stops at the first error and does not invoke an Agent for recovery.
- Stored locator fallback is enabled by default. Optional model-proposed locator
  hints require explicit configuration and the same mechanical checks; they are
  disabled by default. Keep them disabled for model-free replay.

This initial repository contains the runtime and package documentation. Test
suites, experiment runners, benchmark integrations, research reports, and generated
browser data are maintained outside this release.
