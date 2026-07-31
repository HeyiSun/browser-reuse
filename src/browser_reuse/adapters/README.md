# Adapters

`browser.py` contains the first site-neutral browser action vocabulary proven by both the controlled Journey and public EasyAppointments task:

- `Click`
- `Fill`
- `SelectOption`
- CSS and accessible-role targets
- mapping serialization and Playwright execution

`grounding.py` captures the main-document DOM and native accessibility tree,
joins both through Chromium backend node identity, and creates snapshot-local
refs. `generic.py` executes those exact live refs or strictly replays a single
witnessed role/CSS target. Main-document open shadow DOM is supported; iframe,
OOPIF, and closed-shadow controls fail closed in the current slice.

Snapshot refs and durable recipe targets are intentionally separate. A source
action may be executable without being compilable, while fresh-context hard
replay remains the test of whether a witnessed target is actually durable.

Navigation bootstrap and verifier logic remain scenario-specific. Android
implementations remain outside these browser types.
