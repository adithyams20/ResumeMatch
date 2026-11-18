"""
tests/test_config.py — Unit tests for config.py streaming secrets support.
"""

from __future__ import annotations

import os
import pytest
import importlib
import sys
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_env():
    """Save and restore environment state before/after each test."""
    # Save original env state for keys we'll set
    keys_to_clean = [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "LLM_PROVIDER", "OPENAI_MODEL", "ANTHROPIC_MODEL"
    ]
    original_values = {k: os.environ.get(k) for k in keys_to_clean}

    # Clean environment for tests
    for key in keys_to_clean:
        os.environ.pop(key, None)

    yield

    # Restore original environment
    for key, val in original_values.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


# ---------------------------------------------------------------------------
# Helper to reload config module
# ---------------------------------------------------------------------------


def _reload_config(secrets_override: dict | None = None):
    """Force a reload of the config module to pick up new env/secrets state.

    Args:
        secrets_override: Optional dict to provide directly as secrets.
                          If provided, _set_test_secrets will be called.
    """
    # Remove cached module to force reload
    if "config" in sys.modules:
        del sys.modules["config"]

    import config
    importlib.reload(config)

    # If secrets_override is provided, use the test helper
    if secrets_override is not None:
        config._set_test_secrets(secrets_override)

    return config


# ---------------------------------------------------------------------------
# Tests: Environment Variables
# ---------------------------------------------------------------------------


class TestConfigEnvVars:
    """Tests for environment variable loading."""

    def test_env_var_openai_api_key(self, clean_env):
        """OPENAI_API_KEY is read from environment variable."""
        os.environ["OPENAI_API_KEY"] = "sk-test-openai-key"
        config = _reload_config()
        assert config.OPENAI_API_KEY == "sk-test-openai-key"

    def test_env_var_anthropic_api_key(self, clean_env):
        """ANTHROPIC_API_KEY is read from environment variable."""
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"
        config = _reload_config()
        assert config.ANTHROPIC_API_KEY == "sk-ant-test-key"

    def test_env_var_llm_provider(self, clean_env):
        """LLM_PROVIDER is read from environment variable."""
        os.environ["LLM_PROVIDER"] = "anthropic"
        config = _reload_config()
        assert config.LLM_PROVIDER == "anthropic"

    def test_env_var_llm_provider_lowercase(self, clean_env):
        """LLM_PROVIDER is normalized to lowercase."""
        os.environ["LLM_PROVIDER"] = "ANTHROPIC"
        config = _reload_config()
        assert config.LLM_PROVIDER == "anthropic"

    def test_env_var_openai_model(self, clean_env):
        """OPENAI_MODEL is read from environment variable."""
        os.environ["OPENAI_MODEL"] = "gpt-4o"
        config = _reload_config()
        assert config.OPENAI_MODEL == "gpt-4o"

    def test_env_var_anthropic_model(self, clean_env):
        """ANTHROPIC_MODEL is read from environment variable."""
        os.environ["ANTHROPIC_MODEL"] = "claude-3-opus-20240229"
        config = _reload_config()
        assert config.ANTHROPIC_MODEL == "claude-3-opus-20240229"

    def test_default_provider(self, clean_env):
        """LLM_PROVIDER defaults to 'openai' when not set."""
        config = _reload_config()
        assert config.LLM_PROVIDER == "openai"

    def test_default_openai_model(self, clean_env):
        """OPENAI_MODEL defaults to 'gpt-4o-mini' when not set."""
        config = _reload_config()
        assert config.OPENAI_MODEL == "gpt-4o-mini"

    def test_default_anthropic_model(self, clean_env):
        """ANTHROPIC_MODEL defaults to 'claude-3-haiku-20240307' when not set."""
        config = _reload_config()
        assert config.ANTHROPIC_MODEL == "claude-3-haiku-20240307"

    def test_api_keys_default_to_none(self, clean_env):
        """API keys are None when not configured."""
        config = _reload_config()
        assert config.OPENAI_API_KEY is None
        assert config.ANTHROPIC_API_KEY is None

    def test_empty_api_key_from_env(self, clean_env):
        """Empty string API key from env var is treated as missing (None)."""
        os.environ["OPENAI_API_KEY"] = ""
        config = _reload_config()
        # Empty string is treated as missing - returns None
        assert config.OPENAI_API_KEY is None


# ---------------------------------------------------------------------------
# Tests: Streamlit Secrets
# ---------------------------------------------------------------------------


class TestConfigStreamlitSecrets:
    """Tests for Streamlit secrets (st.secrets) loading."""

    def test_st_secrets_direct_key(self, clean_env):
        """Streamlit secrets support direct key lookup."""
        mock_secrets = {
            "OPENAI_API_KEY": "sk-secret-from-st",
            "ANTHROPIC_API_KEY": "sk-ant-secret-from-st",
            "LLM_PROVIDER": "anthropic",
            "OPENAI_MODEL": "gpt-4o",
            "ANTHROPIC_MODEL": "claude-3-opus",
        }

        config = _reload_config(secrets_override=mock_secrets)
        assert config.OPENAI_API_KEY == "sk-secret-from-st"
        assert config.ANTHROPIC_API_KEY == "sk-ant-secret-from-st"
        assert config.LLM_PROVIDER == "anthropic"
        assert config.OPENAI_MODEL == "gpt-4o"
        assert config.ANTHROPIC_MODEL == "claude-3-opus"

    def test_st_secrets_uppercase_keys(self, clean_env):
        """Streamlit secrets support uppercase keys."""
        mock_secrets = {
            "OPENAI_API_KEY": "sk-upper-key",
        }

        config = _reload_config(secrets_override=mock_secrets)
        assert config.OPENAI_API_KEY == "sk-upper-key"

    def test_st_secrets_nested_structure(self, clean_env):
        """Streamlit secrets support nested structure (section)."""
        mock_secrets = {
            "general": {
                "OPENAI_API_KEY": "sk-nested-key",
                "LLM_PROVIDER": "openai",
            },
            "models": {
                "OPENAI_MODEL": "gpt-4-turbo",
                "ANTHROPIC_MODEL": "claude-3-sonnet",
            }
        }

        config = _reload_config(secrets_override=mock_secrets)
        assert config.OPENAI_API_KEY == "sk-nested-key"
        assert config.LLM_PROVIDER == "openai"

    def test_st_secrets_env_takes_precedence(self, clean_env):
        """Environment variables take precedence over st.secrets."""
        mock_secrets = {
            "OPENAI_API_KEY": "sk-from-st-secrets",
        }

        os.environ["OPENAI_API_KEY"] = "sk-from-env"
        config = _reload_config(secrets_override=mock_secrets)
        assert config.OPENAI_API_KEY == "sk-from-env"

    def test_st_secrets_missing_keys_handled_gracefully(self, clean_env):
        """Missing secrets don't crash the application."""
        mock_secrets = {
            # Missing API keys entirely
        }

        config = _reload_config(secrets_override=mock_secrets)
        assert config.OPENAI_API_KEY is None
        assert config.ANTHROPIC_API_KEY is None
        assert config.LLM_PROVIDER == "openai"  # default

    def test_st_secrets_no_streamlit_module(self, clean_env):
        """Works when Streamlit is not installed or not available."""
        # When Streamlit is not available, _try_get_st_secrets returns {}
        config = _reload_config()
        # Should use defaults/env vars
        assert config.LLM_PROVIDER == "openai"
        assert config.OPENAI_API_KEY is None


# ---------------------------------------------------------------------------
# Tests: Missing Keys Graceful Handling
# ---------------------------------------------------------------------------


class TestConfigMissingKeysGraceful:
    """Tests for graceful handling of missing API keys."""

    def test_missing_openai_key_returns_none(self, clean_env):
        """Missing OPENAI_API_KEY returns None without crashing."""
        config = _reload_config()
        assert config.OPENAI_API_KEY is None

    def test_missing_anthropic_key_returns_none(self, clean_env):
        """Missing ANTHROPIC_API_KEY returns None without crashing."""
        config = _reload_config()
        assert config.ANTHROPIC_API_KEY is None

    def test_app_can_check_for_missing_keys(self, clean_env):
        """App can check if keys are missing without crashing."""
        config = _reload_config()

        # App should be able to check key availability
        has_openai = config.OPENAI_API_KEY is not None
        has_anthropic = config.ANTHROPIC_API_KEY is not None
        assert has_openai is False
        assert has_anthropic is False


# ---------------------------------------------------------------------------
# Tests: Helper Functions
# ---------------------------------------------------------------------------


class TestGetSecretHelper:
    """Tests for the _get_secret helper function."""

    def test_get_secret_returns_env_value(self, clean_env):
        """_get_secret returns value from environment if present."""
        from config import _get_secret

        with patch.dict("os.environ", {"TEST_KEY": "test_value"}):
            result = _get_secret("TEST_KEY", secrets={})
            assert result == "test_value"

    def test_get_secret_returns_secret_value(self, clean_env):
        """_get_secret returns value from secrets dict if env not set."""
        from config import _get_secret

        result = _get_secret("TEST_KEY", secrets={"TEST_KEY": "secret_value"})
        assert result == "secret_value"

    def test_get_secret_returns_default(self, clean_env):
        """_get_secret returns default if key not found anywhere."""
        from config import _get_secret

        result = _get_secret("MISSING_KEY", secrets={}, default="default_value")
        assert result == "default_value"

    def test_get_secret_env_precedence(self, clean_env):
        """Environment variables take precedence over secrets."""
        from config import _get_secret

        with patch.dict("os.environ", {"TEST_KEY": "env_value"}):
            result = _get_secret("TEST_KEY", secrets={"TEST_KEY": "secret_value"})
            assert result == "env_value"


# ---------------------------------------------------------------------------
# Tests: No Hardcoded Secrets
# ---------------------------------------------------------------------------


class TestNoHardcodedSecrets:
    """Tests to ensure no secrets are hardcoded in config."""

    def test_no_hardcoded_openai_key(self):
        """OPENAI_API_KEY is not hardcoded to a specific value."""
        # Import fresh config
        config = _reload_config()
        if config.OPENAI_API_KEY:
            # If a key is set, it should not be a common test/demo key
            assert "sk-" not in config.OPENAI_API_KEY or len(config.OPENAI_API_KEY) > 20

    def test_no_hardcoded_anthropic_key(self):
        """ANTHROPIC_API_KEY is not hardcoded to a specific value."""
        config = _reload_config()
        if config.ANTHROPIC_API_KEY:
            if "sk-ant-" in config.ANTHROPIC_API_KEY:
                # If it looks like a key, it should be from env/secrets, not hardcoded
                # Check it's not one of the demo values
                assert config.ANTHROPIC_API_KEY not in ["sk-ant-test", "sk-ant-demo"]


# ---------------------------------------------------------------------------
# Tests: Other Config Values Preserved
# ---------------------------------------------------------------------------


class TestConfigPreservedValues:
    """Tests to ensure other config values are preserved."""

    def test_upload_constraints_preserved(self, clean_env):
        """Upload constraints are preserved."""
        config = _reload_config()
        assert config.MAX_UPLOAD_SIZE_MB == 5
        assert ".pdf" in config.ALLOWED_EXTENSIONS
        assert ".docx" in config.ALLOWED_EXTENSIONS

    def test_paths_preserved(self):
        """Path configuration is preserved."""
        config = _reload_config()
        assert config.DATA_DIR.name == "data"
        assert config.CHROMA_DIR.name == ".chromadb"

    def test_llm_settings_preserved(self, clean_env):
        """LLM settings are preserved."""
        config = _reload_config()
        assert config.LLM_TEMPERATURE == 0.0
        assert config.LLM_MAX_TOKENS == 1024
        assert config.LLM_TIMEOUT == 30
        assert config.LLM_MAX_RETRIES == 3

    def test_nlp_settings_preserved(self):
        """NLP settings are preserved."""
        config = _reload_config()
        assert config.SPACY_MODEL == "en_core_web_sm"
        assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"

    def test_scoring_weights_preserved(self):
        """Scoring weights are preserved."""
        config = _reload_config()
        assert config.WEIGHT_SKILL == 0.40
        assert config.WEIGHT_SEMANTIC == 0.40
        assert config.WEIGHT_EXPERIENCE == 0.20

    def test_section_headers_preserved(self):
        """Section headers are preserved."""
        config = _reload_config()
        assert "experience" in config.SECTION_HEADERS
        assert "skills" in config.SECTION_HEADERS
        assert "education" in config.SECTION_HEADERS