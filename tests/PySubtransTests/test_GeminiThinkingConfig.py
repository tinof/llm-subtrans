from __future__ import annotations

from PySubtrans.Helpers.TestCases import LoggedTestCase
from PySubtrans.Providers.Clients.GeminiClient import GeminiClient
from PySubtrans.SettingsType import SettingsType


class GeminiThinkingConfigTests(LoggedTestCase):
    """Validate how GeminiClient resolves thinking_level vs thinking_budget."""

    def _make_client(self, **extra) -> GeminiClient:
        settings = SettingsType(
            {
                "instructions": "Verify thinking config.",
                "model": "gemini-3.5-flash",
                **extra,
            }
        )
        return GeminiClient(settings)

    def test_thinking_level_takes_precedence(self) -> None:
        """A configured thinking_level produces a level-based ThinkingConfig (Gemini 3.x)."""
        client = self._make_client(thinking_level="low")
        self.assertLoggedIsNotNone("thinking_config", client.thinking_config)
        assert client.thinking_config is not None
        # The SDK normalises the string into the ThinkingLevel enum (e.g. ThinkingLevel.LOW).
        level = getattr(
            client.thinking_config.thinking_level,
            "value",
            client.thinking_config.thinking_level,
        )
        self.assertLoggedEqual("thinking_level", "low", str(level).lower())
        # thinking_budget and thinking_level are mutually exclusive (HTTP 400 if both)
        self.assertLoggedIsNone("thinking_budget", client.thinking_config.thinking_budget)

    def test_default_leaves_dynamic_thinking(self) -> None:
        """Without any thinking setting the config is None (model default dynamic thinking)."""
        client = self._make_client()
        self.assertLoggedIsNone("thinking_config", client.thinking_config)

    def test_budget_path_when_enabled(self) -> None:
        """enable_thinking without a level uses the integer budget (Gemini 2.5.x)."""
        client = self._make_client(enable_thinking=True, thinking_budget=1024)
        self.assertLoggedIsNotNone("thinking_config", client.thinking_config)
        assert client.thinking_config is not None
        self.assertLoggedEqual("thinking_budget", 1024, client.thinking_config.thinking_budget)
        self.assertLoggedIsNone("thinking_level", client.thinking_config.thinking_level)
