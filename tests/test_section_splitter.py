"""
tests/test_section_splitter.py — Unit tests for nlp/section_splitter.py.
"""

from __future__ import annotations

import pytest

from nlp.section_splitter import _detect_section_header, split_sections


# ---------------------------------------------------------------------------
# Sample resumes
# ---------------------------------------------------------------------------

RESUME_WITH_CLEAR_SECTIONS = """\
John Doe — Software Engineer

SUMMARY
Experienced software engineer with 5 years of Python development.

EXPERIENCE
2020 - 2023 | Acme Corp | Backend Engineer
- Built REST APIs
- Managed PostgreSQL databases

SKILLS
Python, FastAPI, PostgreSQL, Docker

EDUCATION
B.Sc. Computer Science, MIT, 2019

PROJECTS
- Personal finance tracker (Python, SQLite)
"""

RESUME_NO_SECTIONS = """\
Jane Smith
I have worked with Python and Java for 3 years.
Built several web applications using React and Node.js.
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSplitSections:
    def test_detects_experience_section(self) -> None:
        sections = split_sections(RESUME_WITH_CLEAR_SECTIONS)
        assert "experience" in sections

    def test_detects_skills_section(self) -> None:
        sections = split_sections(RESUME_WITH_CLEAR_SECTIONS)
        assert "skills" in sections

    def test_detects_education_section(self) -> None:
        sections = split_sections(RESUME_WITH_CLEAR_SECTIONS)
        assert "education" in sections

    def test_detects_projects_section(self) -> None:
        sections = split_sections(RESUME_WITH_CLEAR_SECTIONS)
        assert "projects" in sections

    def test_skills_content_correct(self) -> None:
        sections = split_sections(RESUME_WITH_CLEAR_SECTIONS)
        assert "Python" in sections.get("skills", "")

    def test_experience_content_correct(self) -> None:
        sections = split_sections(RESUME_WITH_CLEAR_SECTIONS)
        assert "REST APIs" in sections.get("experience", "")

    def test_no_sections_falls_back_to_other(self) -> None:
        sections = split_sections(RESUME_NO_SECTIONS)
        # Should not be empty
        assert len(sections) > 0
        # All text should be captured
        combined = " ".join(sections.values())
        assert "Python" in combined

    def test_returns_non_empty_strings(self) -> None:
        sections = split_sections(RESUME_WITH_CLEAR_SECTIONS)
        for label, text in sections.items():
            assert text.strip(), f"Section '{label}' is empty"

    def test_returns_dict(self) -> None:
        result = split_sections(RESUME_WITH_CLEAR_SECTIONS)
        assert isinstance(result, dict)


class TestDetectSectionHeader:
    def _make_map(self) -> dict[str, str]:
        """Build a small keyword→label map for testing."""
        return {
            "experience": "experience",
            "work experience": "experience",
            "skills": "skills",
            "education": "education",
        }

    def test_plain_header(self) -> None:
        result = _detect_section_header("EXPERIENCE", self._make_map())
        assert result == "experience"

    def test_header_with_colon(self) -> None:
        result = _detect_section_header("Skills:", self._make_map())
        assert result == "skills"

    def test_long_line_not_detected(self) -> None:
        long_line = "This is a very long line that is definitely not a section header at all"
        result = _detect_section_header(long_line, self._make_map())
        assert result is None

    def test_none_for_body_text(self) -> None:
        result = _detect_section_header("Built REST APIs using FastAPI.", self._make_map())
        assert result is None

    def test_case_insensitive(self) -> None:
        result = _detect_section_header("education", self._make_map())
        assert result == "education"

