"""
nlp/preprocess.py — Text cleaning, sentence splitting, and spaCy NLP helpers.

Provides a cached spaCy model loader (for use with st.cache_resource)
and utility functions for normalising resume / JD text.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# spaCy model loader
# ---------------------------------------------------------------------------

def load_spacy_model(model_name: str = "en_core_web_sm") -> Any:
    """Load and return a spaCy language model.

    Intended to be wrapped with ``st.cache_resource`` at the call site so
    the model is only loaded once per Streamlit session.

    Args:
        model_name: spaCy model identifier (e.g. ``"en_core_web_sm"``).

    Returns:
        A spaCy ``Language`` object.

    Raises:
        OSError: If the model has not been downloaded. Run
                 ``python -m spacy download en_core_web_sm``.
    """
    import spacy  # local import so module loads even without spaCy installed

    try:
        nlp = spacy.load(model_name)
        logger.info("Loaded spaCy model: %s", model_name)
        return nlp
    except OSError as exc:
        raise OSError(
            f"spaCy model '{model_name}' is not installed. "
            f"Run:  python -m spacy download {model_name}"
        ) from exc


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalise unicode, collapse whitespace, and strip control characters.

    Args:
        text: Raw text extracted from a document.

    Returns:
        Cleaned text suitable for NLP processing.
    """
    # Normalise unicode (e.g., fancy quotes → ASCII)
    text = unicodedata.normalize("NFKD", text)
    # Remove non-printable control characters (except newlines and tabs)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)
    # Replace various dash/bullet characters with hyphens
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219]", "-", text)
    # Collapse horizontal whitespace
    text = re.sub(r"[ \t\u00A0]+", " ", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def split_sentences(text: str, nlp: Any | None = None) -> list[str]:
    """Split text into sentences using spaCy (if available) or heuristics.

    Args:
        text: Input text (cleaned or raw).
        nlp: A loaded spaCy ``Language`` model, or ``None`` to use the
             built-in regex-based fallback.

    Returns:
        List of sentence strings, stripped of leading/trailing whitespace.
    """
    if nlp is not None:
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences

    # Fallback: split on terminal punctuation or newlines
    raw = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in raw if s.strip()]


# ---------------------------------------------------------------------------
# Entity and noun-phrase extraction
# ---------------------------------------------------------------------------

def extract_noun_phrases(text: str, nlp: Any) -> list[str]:
    """Extract noun phrases from text using spaCy's noun chunker.

    Args:
        text: Input text.
        nlp: Loaded spaCy ``Language`` model.

    Returns:
        Deduplicated list of lowercase noun-phrase strings.
    """
    doc = nlp(text)
    phrases = {chunk.text.lower().strip() for chunk in doc.noun_chunks if chunk.text.strip()}
    return sorted(phrases)


def extract_entities(text: str, nlp: Any) -> dict[str, list[str]]:
    """Extract named entities grouped by label.

    Args:
        text: Input text.
        nlp: Loaded spaCy ``Language`` model.

    Returns:
        Dict mapping entity label (e.g. ``"ORG"``, ``"DATE"``) to a list
        of entity text strings.
    """
    doc = nlp(text)
    entities: dict[str, list[str]] = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, []).append(ent.text)
    return entities


# ---------------------------------------------------------------------------
# Years-of-experience extraction
# ---------------------------------------------------------------------------

# Patterns like "5+ years", "3-5 years of experience", "over 7 years"
_YOE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(\d+)\s*\+?\s*(?:to|-)\s*\d*\s*\+?\s*years?\s+(?:of\s+)?experience",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\d+)\s*\+\s*years?\s+(?:of\s+)?experience",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:minimum|at\s+least|over|more\s+than)\s+(\d+)\s*\+?\s*years?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\d+)\s*years?\s+(?:of\s+)?(?:relevant\s+)?experience",
        re.IGNORECASE,
    ),
]


def extract_required_experience(text: str) -> float:
    """Heuristically extract the minimum years of experience from JD text.

    Args:
        text: Job description text.

    Returns:
        Required years as a float, or ``0.0`` if not found.
    """
    for pattern in _YOE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except (IndexError, ValueError):
                continue
    return 0.0


def extract_resume_experience(text: str) -> float:
    """Heuristically estimate total years of experience from resume text.

    Looks for date ranges (e.g. "2018 – 2023") and sums the durations.

    Args:
        text: Resume text.

    Returns:
        Estimated total years as a float.
    """
    import datetime

    current_year = datetime.date.today().year

    # Match year ranges like "2018 - 2023" or "2020 – Present"
    range_pattern = re.compile(
        r"(\d{4})\s*[-–—]\s*(\d{4}|present|current|now)",
        re.IGNORECASE,
    )
    total_years = 0.0
    for match in range_pattern.finditer(text):
        start = int(match.group(1))
        end_raw = match.group(2).lower()
        end = current_year if end_raw in {"present", "current", "now"} else int(end_raw)
        if 1970 <= start <= current_year and start <= end <= current_year + 1:
            total_years += end - start

    return round(total_years, 1)

