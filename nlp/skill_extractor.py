"""
nlp/skill_extractor.py — Skill extraction using a seed list + spaCy noun chunks.

Combines three signals:
1. Exact match against the skills seed list (data/skills.txt).
2. Fuzzy match (rapidfuzz) for minor spelling variations.
3. spaCy noun-phrase chunks that overlap with known skills.

Alias normalisation maps common alternate spellings to canonical seed-list
entries so that e.g. "ReactJS" and "React" both resolve to "React".
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
# Alias map  (alternate spelling / abbreviation → canonical seed-list name)
# ---------------------------------------------------------------------------

# Keys are lowercased; values must appear verbatim in data/skills.txt.
_ALIASES: dict[str, str] = {
    # JavaScript ecosystem
    "reactjs": "React",
    "react.js": "React",
    "react js": "React",
    "vuejs": "Vue",
    "vue.js": "Vue",
    "angularjs": "Angular",
    "angular.js": "Angular",
    "nextjs": "Next.js",
    "next js": "Next.js",
    "nuxtjs": "Nuxt.js",
    "nuxt js": "Nuxt.js",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node js": "Node.js",
    "expressjs": "Express",
    "express.js": "Express",
    "nestjs": "NestJS",
    "nest.js": "NestJS",
    "sveltejs": "Svelte",
    "svelte.js": "Svelte",
    # Python
    "py": "Python",
    # TypeScript
    "ts": "TypeScript",
    # Databases
    "postgres": "PostgreSQL",
    "mongo": "MongoDB",
    "elastic": "Elasticsearch",
    # Cloud
    "amazon web services": "AWS",
    "google cloud platform": "GCP",
    "google cloud": "GCP",
    "microsoft azure": "Azure",
    # CI/CD
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    # Other
    "fast api": "FastAPI",
    "django rest framework": "Django",
    "drf": "Django",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "huggingface": "Hugging Face",
    "hf": "Hugging Face",
}


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

    Applies the alias map so alternate spellings (e.g. "ReactJS", "NodeJS")
    resolve to the canonical seed-list entry.

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

    # Build a lookup: lowercase → original casing (seed list + aliases)
    lower_to_original: dict[str, str] = {s.lower(): s for s in seed_skills}
    # Merge alias targets that are present in the seed list
    alias_lower_to_original: dict[str, str] = {}
    for alias_lc, canonical in _ALIASES.items():
        if canonical in seed_skills or canonical.lower() in lower_to_original:
            resolved = lower_to_original.get(canonical.lower(), canonical)
            alias_lower_to_original[alias_lc] = resolved
    # Combined lookup (seed takes precedence over alias for same key)
    full_lookup = {**alias_lower_to_original, **lower_to_original}

    # All keys to match against (seed names + alias keys)
    all_keys: list[str] = sorted(full_lookup.keys(), key=len, reverse=True)

    matched: set[str] = set()

    # --- 1. Word-boundary match (seed + aliases) ---
    # Single-letter skills (e.g. "C", "R") are matched case-sensitively and
    # only as whole words -- never as a substring of "C++", "C#", "R2D2", etc.
    text_lower = text.lower()
    for key in all_keys:
        if len(key) == 1:
            pattern = re.compile(
                r"(?<![A-Za-z0-9+#-])" + re.escape(key) + r"(?![A-Za-z0-9+#-])"
            )
            if pattern.search(text):
                matched.add(full_lookup[key])
        else:
            pattern = re.compile(
                r"(?<!\w)" + re.escape(key) + r"(?!\w)", re.IGNORECASE
            )
            if pattern.search(text_lower):
                matched.add(full_lookup[key])

    # --- 2. Fuzzy match on individual tokens / short n-grams ---
    tokens = _ngrams(text, max_n=4)
    for token in tokens:
        if len(token) < 2:
            continue
        token_lc = token.lower()
        # Check aliases first (exact)
        if token_lc in alias_lower_to_original:
            matched.add(alias_lower_to_original[token_lc])
            continue
        result = process.extractOne(
            token_lc,
            seed_lower,
            scorer=fuzz.ratio,
            score_cutoff=fuzzy_threshold,
        )
        if result:
            matched.add(lower_to_original[result[0]])

    # --- 3. spaCy noun-chunk overlap ---
    if nlp is not None:
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            chunk_lower = chunk.text.lower().strip()
            if chunk_lower in full_lookup:
                matched.add(full_lookup[chunk_lower])
            else:
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

    Uses fuzzy matching AND the alias map so that e.g. a resume containing
    "React" will match a JD skill of "ReactJS" (and vice-versa).

    Args:
        resume_skills: Skills extracted from the resume.
        jd_skills: Skills extracted from the job description.
        fuzzy_threshold: Minimum rapidfuzz score to count as a match.

    Returns:
        Tuple of (matched_skills, missing_skills) — both lists contain
        the original JD skill strings.
    """
    # Build expanded resume set: canonical name + all alias targets
    resume_lower_set: set[str] = set()
    for s in resume_skills:
        resume_lower_set.add(s.lower())
        # Also add any canonical form from the alias map
        for alias_lc, canonical in _ALIASES.items():
            if s.lower() == canonical.lower() or s.lower() == alias_lc:
                resume_lower_set.add(canonical.lower())
                resume_lower_set.add(alias_lc)

    resume_lower = list(resume_lower_set)
    matched: list[str] = []
    missing: list[str] = []

    for jd_skill in jd_skills:
        jd_lower = jd_skill.lower()
        # Also check under alias expansions of the JD skill
        jd_variants = {jd_lower}
        canonical = _ALIASES.get(jd_lower)
        if canonical:
            jd_variants.add(canonical.lower())
        # Check reverse: jd_skill is a canonical, look for aliases pointing to it
        for alias_lc, can in _ALIASES.items():
            if can.lower() == jd_lower:
                jd_variants.add(alias_lc)

        found = False
        for variant in jd_variants:
            result = process.extractOne(
                variant,
                resume_lower,
                scorer=fuzz.ratio,
                score_cutoff=fuzzy_threshold,
            )
            if result is not None:
                found = True
                break

        if found:
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

