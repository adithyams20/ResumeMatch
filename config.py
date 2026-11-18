"""
config.py — Central configuration for ResumeMatch.

All tunable knobs (weights, model names, paths, thresholds) live here.
Reads environment variables via python-dotenv or Streamlit secrets.
Prefers environment variables, falls back to st.secrets.
Never hardcodes API keys.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Upload Constraints
# ---------------------------------------------------------------------------
MAX_UPLOAD_SIZE_MB: int = 5
ALLOWED_EXTENSIONS: list[str] = [".pdf", ".docx"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).parent
DATA_DIR: Path = BASE_DIR / "data"
TMP_UPLOAD_DIR: Path = BASE_DIR / "tmp_uploads"
SKILLS_FILE: Path = DATA_DIR / "skills.txt"
CHROMA_DIR: Path = BASE_DIR / ".chromadb"

# ---------------------------------------------------------------------------
# Load .env (only if the file exists; does not override already-set vars)
# ---------------------------------------------------------------------------
load_dotenv(BASE_DIR / ".env", override=False)


# ---------------------------------------------------------------------------
# Streamlit secrets support
# ---------------------------------------------------------------------------
_st_secrets: dict[str, Any] = {}  # Will be populated at module load


def _try_get_st_secrets() -> dict[str, Any]:
    """Try to import st.secrets from Streamlit Community Cloud.

    Returns an empty dict if Streamlit is not available or secrets are not set.
    This function is safe to call even when running outside of Streamlit.
    """
    try:
        import streamlit as st

        if hasattr(st, "secrets") and isinstance(st.secrets, dict):
            return dict(st.secrets)
    except ImportError:
        pass
    except Exception:
        # Any other error accessing st.secrets (e.g., not on Streamlit Cloud)
        pass
    return {}


def _get_secret(key: str, secrets: dict[str, Any] | None = None, default: str | None = None) -> str | None:
    """Get a secret value, preferring env vars over st.secrets.

    Args:
        key: The secret key to look up.
        secrets: Pre-fetched secrets dict (for testability). If None, will try to fetch from st.secrets.
        default: Default value if not found anywhere.

    Returns:
        The secret value or default/None if not found.
    """
    # First, check environment variables (highest priority)
    env_value = os.getenv(key)
    if env_value is not None and env_value != "":
        return env_value

    # Then, check st.secrets
    if secrets is None:
        secrets = _st_secrets

    # Streamlit secrets use uppercase keys, but can also be in sections
    # Try direct key first, then try nested structure
    if key in secrets:
        return secrets.get(key)

    # Try uppercase version
    upper_key = key.upper()
    if upper_key in secrets:
        return secrets.get(upper_key)

    # Check if it's in a nested structure like st.secrets["general"]["KEY"]
    for section in secrets.values():
        if isinstance(section, dict) and key in section:
            return section.get(key)
        if isinstance(section, dict) and upper_key in section:
            return section.get(upper_key)

    return default


# Fetch secrets once at module load time
_st_secrets = _try_get_st_secrets()


# ---------------------------------------------------------------------------
# Test helper: Allow tests to override secrets after module load
# ---------------------------------------------------------------------------
def _set_test_secrets(secrets: dict[str, Any]) -> None:
    """Override the internal secrets dict for testing.

    This allows tests to provide mock secrets without needing to patch
    the streamlit module directly.
    """
    global _st_secrets
    _st_secrets = secrets

    # Also refresh the config values
    _refresh_config_values()


def _refresh_config_values() -> None:
    """Re-evaluate config values from the updated secrets."""
    global LLM_PROVIDER, OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENAI_MODEL, ANTHROPIC_MODEL
    LLM_PROVIDER = _get_secret("LLM_PROVIDER", _st_secrets, default="openai").lower()  # "openai" | "anthropic"
    OPENAI_API_KEY = _get_secret("OPENAI_API_KEY", _st_secrets)
    ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY", _st_secrets)
    OPENAI_MODEL = _get_secret("OPENAI_MODEL", _st_secrets, default="gpt-4o-mini")
    ANTHROPIC_MODEL = _get_secret("ANTHROPIC_MODEL", _st_secrets, default="claude-3-haiku-20240307")


# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = _get_secret("LLM_PROVIDER", _st_secrets, default="openai").lower()  # "openai" | "anthropic"

# API keys - can be None if not configured
OPENAI_API_KEY: str | None = _get_secret("OPENAI_API_KEY", _st_secrets)
ANTHROPIC_API_KEY: str | None = _get_secret("ANTHROPIC_API_KEY", _st_secrets)

# Model names - have defaults if not configured
OPENAI_MODEL: str = _get_secret("OPENAI_MODEL", _st_secrets, default="gpt-4o-mini")
ANTHROPIC_MODEL: str = _get_secret("ANTHROPIC_MODEL", _st_secrets, default="claude-3-haiku-20240307")

LLM_TEMPERATURE: float = 0.0
LLM_MAX_TOKENS: int = 1024
LLM_TIMEOUT: int = 30          # seconds
LLM_MAX_RETRIES: int = 3

# ---------------------------------------------------------------------------
# NLP / spaCy
# ---------------------------------------------------------------------------
SPACY_MODEL: str = "en_core_web_sm"

# ---------------------------------------------------------------------------
# Embedding model (sentence-transformers)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# RAG / ChromaDB
# ---------------------------------------------------------------------------
CHROMA_COLLECTION: str = "resume_chunks"
CHUNK_MAX_WORDS: int = 150     # max words per chunk
TOP_K_RETRIEVAL: int = 3       # chunks retrieved per JD requirement

# ---------------------------------------------------------------------------
# Skill matching
# ---------------------------------------------------------------------------
FUZZY_THRESHOLD: int = 90      # rapidfuzz score threshold (0-100)

# ---------------------------------------------------------------------------
# Scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHT_SKILL: float = 0.40
WEIGHT_SEMANTIC: float = 0.40
WEIGHT_EXPERIENCE: float = 0.20

# ---------------------------------------------------------------------------
# Section headers (used by section_splitter)
# ---------------------------------------------------------------------------
SECTION_HEADERS: dict[str, list[str]] = {
    "experience": [
        "experience", "work experience", "professional experience",
        "employment", "work history", "career history",
    ],
    "skills": [
        "skills", "technical skills", "core competencies",
        "technologies", "tools", "expertise",
    ],
    "education": [
        "education", "academic background", "qualifications",
        "degrees", "certifications",
    ],
    "projects": [
        "projects", "personal projects", "open source",
        "side projects", "portfolio",
    ],
    "summary": [
        "summary", "objective", "profile", "about me", "overview",
    ],
}

