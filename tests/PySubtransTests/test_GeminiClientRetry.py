from __future__ import annotations

from unittest.mock import patch

from google.genai import errors as genai_errors

from PySubtrans.Helpers.TestCases import LoggedTestCase
from PySubtrans.Providers.Clients.GeminiClient import GeminiClient
from PySubtrans.SettingsType import SettingsType
from PySubtrans.SubtitleError import TranslationImpossibleError
from PySubtrans.TranslationPrompt import TranslationPrompt
from PySubtrans.TranslationRequest import TranslationRequest


def _api_error(code: int) -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"message": f"error {code}"}})


class GeminiClientRetryTests(LoggedTestCase):
    """Validate GeminiClient retry classification and client reuse."""

    def _make_client(self, **extra) -> GeminiClient:
        settings = SettingsType(
            {
                "instructions": "Verify retry behaviour.",
                "model": "gemini-3.1-flash-lite",
                "max_retries": 3,
                "backoff_time": 0.0,
                **extra,
            }
        )
        return GeminiClient(settings)

    def _make_request(self) -> TranslationRequest:
        prompt = TranslationPrompt("Translate this", False)
        prompt.content = "1. Hello"
        return TranslationRequest(prompt, None)

    def test_retries_transient_429_then_succeeds(self) -> None:
        """A 429 APIError is retried and a later success is returned."""
        client = self._make_client()
        responses = [_api_error(429), _api_error(503), {"text": "translated"}]

        def fake_response(request, temperature):
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        with (
            patch.object(GeminiClient, "_get_gemini_response", side_effect=fake_response),
            patch("PySubtrans.Providers.Clients.GeminiClient.time.sleep") as mock_sleep,
        ):
            result = client._send_messages(self._make_request(), temperature=0.0)

        self.assertLoggedEqual("result", {"text": "translated"}, result)
        self.assertLoggedEqual("sleep_count", 2, mock_sleep.call_count)

    def test_fails_fast_on_non_retryable_error(self) -> None:
        """A 404 APIError raises immediately without retrying."""
        client = self._make_client()

        with (
            patch.object(
                GeminiClient, "_get_gemini_response", side_effect=_api_error(404)
            ) as mock_response,
            patch("PySubtrans.Providers.Clients.GeminiClient.time.sleep") as mock_sleep,
        ):
            with self.assertRaises(TranslationImpossibleError):
                client._send_messages(self._make_request(), temperature=0.0)

        self.assertLoggedEqual("attempts", 1, mock_response.call_count)
        self.assertLoggedEqual("sleep_count", 0, mock_sleep.call_count)

    def test_gives_up_after_max_retries(self) -> None:
        """Persistent 429s raise TranslationImpossibleError after max_retries."""
        client = self._make_client()

        with (
            patch.object(
                GeminiClient, "_get_gemini_response", side_effect=_api_error(429)
            ) as mock_response,
            patch("PySubtrans.Providers.Clients.GeminiClient.time.sleep"),
        ):
            with self.assertRaises(TranslationImpossibleError):
                client._send_messages(self._make_request(), temperature=0.0)

        self.assertLoggedEqual("attempts", 4, mock_response.call_count)

    def test_client_is_created_once_and_reused(self) -> None:
        """_get_client creates the underlying genai client once and caches it."""
        client = self._make_client(api_key="test-key")

        with patch.object(
            GeminiClient, "_create_client", return_value=object()
        ) as mock_create:
            first = client._get_client()
            second = client._get_client()

        self.assertLoggedEqual("create_calls", 1, mock_create.call_count)
        self.assertLoggedIs("same_instance", first, second)
