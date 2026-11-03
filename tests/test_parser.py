"""
tests/test_parser.py — Unit tests for parsers/resume_parser.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from parsers.resume_parser import _clean_text, _parse_txt, parse_resume


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RESUME = """\
John Doe
Software Engineer

EXPERIENCE
2020 - Present | Acme Corp | Senior Python Developer
- Built REST APIs with FastAPI and PostgreSQL.
- Reduced latency by 40% using Redis caching.

SKILLS
Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes

EDUCATION
B.Sc. Computer Science — MIT, 2019
"""


@pytest.fixture()
def txt_resume_file(tmp_path: Path) -> Path:
    """Write a sample TXT resume to a temp file and return the path."""
    f = tmp_path / "resume.txt"
    f.write_text(SAMPLE_RESUME, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseResumeTxt:
    def test_returns_nonempty_string(self, txt_resume_file: Path) -> None:
        text = parse_resume(txt_resume_file)
        assert isinstance(text, str)
        assert len(text) > 50

    def test_contains_expected_content(self, txt_resume_file: Path) -> None:
        text = parse_resume(txt_resume_file)
        assert "John Doe" in text
        assert "FastAPI" in text

    def test_whitespace_is_normalised(self, txt_resume_file: Path) -> None:
        text = parse_resume(txt_resume_file)
        assert "  " not in text  # no double spaces


class TestParseResumeErrors:
    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "resume.odt"
        bad_file.write_text("content", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file format"):
            parse_resume(bad_file)

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_resume("/nonexistent/path/resume.txt")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.txt"
        empty.write_text("   ", encoding="utf-8")
        with pytest.raises(ValueError, match="empty or is a scanned"):
            parse_resume(empty)


class TestCleanText:
    def test_collapses_whitespace(self) -> None:
        assert _clean_text("hello   world") == "hello world"

    def test_strips_null_bytes(self) -> None:
        assert "\x00" not in _clean_text("hello\x00world")

    def test_collapses_excessive_newlines(self) -> None:
        text = "line1\n\n\n\n\nline2"
        result = _clean_text(text)
        assert result.count("\n") <= 2

