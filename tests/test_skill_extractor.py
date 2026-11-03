"""
tests/test_skill_extractor.py — Unit tests for nlp/skill_extractor.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nlp.skill_extractor import _ngrams, extract_skills, load_seed_skills, match_skills


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_skills_file(tmp_path: Path) -> Path:
    """Create a small temporary skills file for testing."""
    skills = [
        "Python",
        "JavaScript",
        "Docker",
        "Kubernetes",
        "PostgreSQL",
        "React",
        "FastAPI",
        "Machine Learning",
    ]
    f = tmp_path / "skills.txt"
    f.write_text("\n".join(skills), encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Tests: load_seed_skills
# ---------------------------------------------------------------------------

class TestLoadSeedSkills:
    def test_loads_correct_count(self, tmp_skills_file: Path) -> None:
        skills = load_seed_skills(tmp_skills_file)
        assert len(skills) == 8

    def test_returns_list_of_strings(self, tmp_skills_file: Path) -> None:
        skills = load_seed_skills(tmp_skills_file)
        assert all(isinstance(s, str) for s in skills)

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        skills = load_seed_skills(tmp_path / "nonexistent.txt")
        assert skills == []


# ---------------------------------------------------------------------------
# Tests: extract_skills
# ---------------------------------------------------------------------------

class TestExtractSkills:
    def test_exact_match(self, tmp_skills_file: Path) -> None:
        text = "We use Python and Docker for our services."
        skills = extract_skills(text, skills_file=tmp_skills_file)
        assert "Python" in skills
        assert "Docker" in skills

    def test_case_insensitive(self, tmp_skills_file: Path) -> None:
        text = "Experience with PYTHON and kubernetes required."
        skills = extract_skills(text, skills_file=tmp_skills_file)
        assert "Python" in skills
        assert "Kubernetes" in skills

    def test_fuzzy_match(self, tmp_skills_file: Path) -> None:
        # "Pythonn" should fuzzy-match "Python"
        text = "Strong Pythonn skills needed."
        skills = extract_skills(text, skills_file=tmp_skills_file, fuzzy_threshold=85)
        assert "Python" in skills

    def test_multiword_skill(self, tmp_skills_file: Path) -> None:
        text = "Experience in machine learning and AI is required."
        skills = extract_skills(text, skills_file=tmp_skills_file)
        assert "Machine Learning" in skills

    def test_no_skills_in_random_text(self, tmp_skills_file: Path) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        skills = extract_skills(text, skills_file=tmp_skills_file)
        assert len(skills) == 0

    def test_returns_sorted_list(self, tmp_skills_file: Path) -> None:
        text = "Python, Docker, and React are required."
        skills = extract_skills(text, skills_file=tmp_skills_file)
        assert skills == sorted(skills)


# ---------------------------------------------------------------------------
# Tests: match_skills
# ---------------------------------------------------------------------------

class TestMatchSkills:
    def test_all_matched(self) -> None:
        resume = ["Python", "Docker", "React"]
        jd = ["Python", "Docker"]
        matched, missing = match_skills(resume, jd)
        assert set(matched) == {"Python", "Docker"}
        assert missing == []

    def test_some_missing(self) -> None:
        resume = ["Python"]
        jd = ["Python", "Kubernetes"]
        matched, missing = match_skills(resume, jd)
        assert "Python" in matched
        assert "Kubernetes" in missing

    def test_fuzzy_matching(self) -> None:
        resume = ["PostgreSQL"]
        jd = ["Postgres"]
        matched, missing = match_skills(resume, jd, fuzzy_threshold=80)
        assert "Postgres" in matched

    def test_empty_jd(self) -> None:
        matched, missing = match_skills(["Python"], [])
        assert matched == []
        assert missing == []


# ---------------------------------------------------------------------------
# Tests: _ngrams
# ---------------------------------------------------------------------------

class TestNgrams:
    def test_unigrams(self) -> None:
        result = _ngrams("hello world", max_n=1)
        assert "hello" in result
        assert "world" in result

    def test_bigrams(self) -> None:
        result = _ngrams("machine learning", max_n=2)
        assert "machine learning" in result

    def test_empty_string(self) -> None:
        assert _ngrams("", max_n=3) == []

