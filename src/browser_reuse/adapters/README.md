# Adapters

`browser.py` contains the first site-neutral browser action vocabulary proven by both the controlled Journey and public EasyAppointments task:

- `Click`
- `Fill`
- `SelectOption`
- CSS and accessible-role targets
- mapping serialization and Playwright execution

Observation extraction, navigation setup, waits, and verifier logic remain scenario-specific. Android implementations remain outside these browser types.
