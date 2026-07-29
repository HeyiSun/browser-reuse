# Adapters

`browser.py` contains the first site-neutral browser action vocabulary proven by both the controlled Journey and public EasyAppointments task:

- `Click`
- `Fill`
- `SelectOption`
- CSS and accessible-role targets
- mapping serialization and Playwright execution

`aria.py` contains the site-neutral `GenericBrowserAdapter` qualified by two
public benchmark sites. It converts Playwright AI-ARIA observations into unique,
actionable controls with recordable role/CSS targets, executes the same typed
browser actions, and waits for a bounded stable public state. Fresh-context hard
replay—not the adapter's naming—is the test of whether a recorded target is
actually durable.

Navigation bootstrap and verifier logic remain scenario-specific. Android
implementations remain outside these browser types.
