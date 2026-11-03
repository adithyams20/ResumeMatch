"""
config.py — Central configuration for ResumeMatch.

All tunable knobs (weights, model names, paths, thresholds) live here.
Reads environment variables via python-dotenv; never hardcodes secrets.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).parent
DATA_DIR: Path = BASE_DIR / "data"
SKILLS_FILE: Path = DATA_DIR / "skills.txt"
CHROMA_DIR: Path = BASE_DIR / ".chromadb"

# ---------------------------------------------------------------------------
# Load .env (only if the file exists; does not override already-set vars)
# ---------------------------------------------------------------------------
load_dotenv(BASE_DIR / ".env", override=False)

# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()  # "openai" | "anthropic"
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")

OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")

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

