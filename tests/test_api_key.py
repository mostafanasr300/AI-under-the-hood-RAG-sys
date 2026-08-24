"""
CI Gate: LLM API Key Validation Test
=====================================
Validates that the Groq API key (grog env var) is:
  1. Present in the environment
  2. Valid — by making a minimal, real API call to openai/gpt-oss-120b
  3. Authorized for the correct model

This test runs as a REQUIRED gate before the Docker image is built and
pushed to GHCR. If the key is missing or rejected, the whole CI pipeline
fails early and no image is produced.

This test is skipped gracefully in offline / mock environments.
"""

import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL = "openai/gpt-oss-120b"


# ============================================================
# HELPER
# ============================================================

def _get_groq_key() -> str | None:
    """Return the Groq API key from the environment, or None."""
    return os.environ.get("grog") or os.environ.get("GROQ_API_KEY")


# ============================================================
# 1. PRESENCE CHECK
# ============================================================

class TestApiKeyPresence:
    """Validates that the API key env var is configured at all."""

    def test_groq_api_key_is_set(self):
        """The 'grog' or GROQ_API_KEY environment variable must be present."""
        key = _get_groq_key()
        assert key is not None, (
            "GROQ_API_KEY (env: 'grog') is not set. "
            "Add it as a GitHub Actions secret named GROQ_API_KEY."
        )

    def test_groq_api_key_not_empty(self):
        """The API key must not be an empty string."""
        key = _get_groq_key()
        assert key and key.strip(), (
            "GROQ_API_KEY (env: 'grog') is set but empty. "
            "Ensure the secret has a valid value."
        )

    def test_groq_api_key_has_minimum_length(self):
        """Groq API keys are typically 50+ characters — catch obvious placeholders."""
        key = _get_groq_key()
        assert key and len(key.strip()) >= 40, (
            f"GROQ_API_KEY looks too short ({len(key or '')} chars). "
            "This is likely a placeholder, not a real key."
        )


# ============================================================
# 2. LIVE CONNECTIVITY CHECK
# ============================================================

class TestApiKeyLiveValidation:
    """
    Makes a real, minimal API call to verify the key is accepted
    and the target model is accessible.

    Skipped automatically if the key is absent (caught by presence tests).
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_key(self):
        """Skip this class entirely when no key is available."""
        if not _get_groq_key():
            pytest.skip("GROQ_API_KEY not set — skipping live validation.")

    def test_model_reachable_and_key_accepted(self):
        """
        Call openai/gpt-oss-120b with a single-token prompt.
        A 200 response confirms both key validity and model availability.
        """
        from groq import Groq, AuthenticationError, NotFoundError, PermissionDeniedError

        client = Groq(api_key=_get_groq_key())

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,          # Absolute minimum — fast & cheap
                temperature=0.0,
            )
        except AuthenticationError as exc:
            pytest.fail(
                f"GROQ_API_KEY was REJECTED by the Groq API.\n"
                f"Check that the secret is correct and not expired.\n"
                f"Details: {exc}"
            )
        except NotFoundError as exc:
            pytest.fail(
                f"Model '{MODEL}' was NOT FOUND on Groq.\n"
                f"The model name may have changed or is not available on your plan.\n"
                f"Details: {exc}"
            )
        except PermissionDeniedError as exc:
            pytest.fail(
                f"GROQ_API_KEY does not have permission to use '{MODEL}'.\n"
                f"Details: {exc}"
            )

        # Validate the response shape
        assert response.choices, "API returned an empty choices list."
        assert response.model, "API response missing model field."

    def test_response_model_matches_expected(self):
        """
        The model returned in the response should match what we requested.
        Guards against silent model routing / aliasing issues.
        """
        from groq import Groq

        client = Groq(api_key=_get_groq_key())
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
        )
        returned_model = response.model or ""
        assert MODEL in returned_model or returned_model in MODEL, (
            f"Requested model '{MODEL}' but got '{returned_model}' in response. "
            "Possible routing mismatch — verify the model name."
        )
