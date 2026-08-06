"""Minimal Playwright browser ownership with fresh per-attempt contexts."""

from __future__ import annotations


class BrowserSession:
    """Own one fresh browser context and page; closing is idempotent."""

    def __init__(self, context, page) -> None:
        self.context = context
        self.page = page
        self._closed = False

    def close(self) -> None:
        """Close the owned context once."""

        if not self._closed:
            self.context.close()
            self._closed = True

    def __enter__(self) -> BrowserSession:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


class PlaywrightRuntime:
    """Own Playwright and Chromium while creating isolated sessions."""

    def __init__(self, *, headless: bool = True) -> None:
        self._headless = headless
        self._playwright = None
        self._browser = None

    def start(self) -> PlaywrightRuntime:
        """Start Chromium once and clean up Playwright if launch fails."""

        if self._browser is not None:
            return self
        self._playwright = _start_playwright()
        try:
            self._browser = self._playwright.chromium.launch(headless=self._headless)
        except Exception:
            self._playwright.stop()
            self._playwright = None
            raise
        return self

    def new_session(
        self,
        *,
        timezone_id: str = "UTC",
        locale: str = "en-US",
        default_timeout_ms: int = 2_000,
    ) -> BrowserSession:
        """Create a fresh context and page with deterministic defaults."""

        if self._browser is None:
            raise RuntimeError("PlaywrightRuntime has not been started")
        context = self._browser.new_context(
            viewport={"width": 1100, "height": 800},
            locale=locale,
            timezone_id=timezone_id,
        )
        try:
            page = context.new_page()
            page.set_default_timeout(default_timeout_ms)
        except Exception:
            context.close()
            raise
        return BrowserSession(context, page)

    def stop(self) -> None:
        """Close Chromium, then Playwright, and allow repeated calls."""

        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    def __enter__(self) -> PlaywrightRuntime:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.stop()


def _start_playwright():
    """Import Playwright lazily so it remains an optional runtime dependency."""

    from playwright.sync_api import sync_playwright

    return sync_playwright().start()
