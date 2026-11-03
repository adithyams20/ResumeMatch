"""
nlp/skill_extractor.py — Skill extraction using a seed list + spaCy noun chunks.

Combines three signals:
1. Exact match against the skills seed list (data/skills.txt).
2. Fuzzy match (rapidfuzz) for minor spelling variations.
3. spaCy noun-phrase chunks that overlap with known skills.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from config import FUZZY_THRESHOLD, SKILLS_FILE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed-skill loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_seed_skills(skills_file: str | Path = SKILLS_FILE) -> list[str]:
    """Load the skill seed list from a plain-text file (one skill per line).

    Args:
        skills_file: Path to the skills text file.

    Returns:
        List of skill strings, normalised to lowercase.
    """
    path = Path(skills_file)
    if not path.exists():
        logger.warning("Skills file not found: %s", path)
        return []
    skills = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    logger.info("Loaded %d seed skills from %s", len(skills), path)
    return skills


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_skills(
    text: str,
    nlp: Any | None = None,
    skills_file: str | Path = SKILLS_FILE,
    fuzzy_threshold: int = FUZZY_THRESHOLD,
) -> list[str]:
    """Extract skills from *text* using seed-list matching + spaCy noun chunks.

    Args:
        text: Resume or job description text.
        nlp: Optional loaded spaCy model for noun-chunk fallback.
        skills_file: Path to the seed skills file.
        fuzzy_threshold: Minimum rapidfuzz score (0–100) to accept a match.

    Returns:
        Deduplicated sorted list of matched skill strings (original casing
        from the seed list).
    """
    seed_skills = load_seed_skills(skills_file)
    seed_lower = [s.lower() for s in seed_skills]

    # Build a lookup: lowercase → original casing
    lower_to_original: dict[str, str] = {
        s.lower(): s for s in seed_skills
    }

    matched: set[str] = set()

    # --- 1. Exact / case-insensitive word-boundary match ---
    text_lower = text.lower()
    for skill_lower, skill_orig in lower_to_original.items():
        pattern = re.compile(
            r"(?<!\w)" + re.escape(skill_lower) + r"(?!\w)", re.IGNORECASE
        )
        if pattern.search(text_lower):
            matched.add(skill_orig)

    # --- 2. Fuzzy match on individual tokens / short n-grams ---
    # Tokenise the text into words and short n-grams (up to 4 words)
    tokens = _ngrams(text, max_n=4)
    for token in tokens:
        if len(token) < 2:
            continue
        result = process.extractOne(
            token.lower(),
            seed_lower,
            scorer=fuzz.ratio,
            score_cutoff=fuzzy_threshold,
        )
        if result:
            best_match_lower = result[0]
            matched.add(lower_to_original[best_match_lower])

    # --- 3. spaCy noun-chunk overlap ---
    if nlp is not None:
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            chunk_lower = chunk.text.lower().strip()
            if chunk_lower in lower_to_original:
                matched.add(lower_to_original[chunk_lower])
            else:
                # Fuzzy check on the chunk
                result = process.extractOne(
                    chunk_lower,
                    seed_lower,
                    scorer=fuzz.ratio,
                    score_cutoff=fuzzy_threshold,
                )
                if result:
                    matched.add(lower_to_original[result[0]])

    return sorted(matched)


def match_skills(
    resume_skills: list[str],
    jd_skills: list[str],
    fuzzy_threshold: int = FUZZY_THRESHOLD,
) -> tuple[list[str], list[str]]:
    """Compare resume skills against JD skills.

    Args:
        resume_skills: Skills extracted from the resume.
        jd_skills: Skills extracted from the job description.
        fuzzy_threshold: Minimum rapidfuzz score to count as a match.

    Returns:
        Tuple of (matched_skills, missing_skills) — both lists contain
        the original JD skill strings.
    """
    resume_lower = [s.lower() for s in resume_skills]
    matched: list[str] = []
    missing: list[str] = []

    for jd_skill in jd_skills:
        jd_lower = jd_skill.lower()
        result = process.extractOne(
            jd_lower,
            resume_lower,
            scorer=fuzz.ratio,
            score_cutoff=fuzzy_threshold,
        )
        if result is not None:
            matched.append(jd_skill)
        else:
            missing.append(jd_skill)

    return matched, missing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ngrams(text: str, max_n: int = 4) -> list[str]:
    """Generate all word n-grams (1 ≤ n ≤ max_n) from *text*.

    Args:
        text: Input string.
        max_n: Maximum n-gram size.

    Returns:
        List of n-gram strings.
    """
    words = re.findall(r"\b\w[\w.+#-]*\b", text)
    result: list[str] = []
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            result.append(" ".join(words[i: i + n]))
    return result

