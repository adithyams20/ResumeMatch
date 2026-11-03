"""
nlp/section_splitter.py — Split resume text into labelled sections.

Uses heuristic regex matching on common section-header keywords
(configured in config.SECTION_HEADERS). Falls back to treating the
entire text as the "other" section if no headers are detected.
"""

from __future__ import annotations

import logging
import re

from config import SECTION_HEADERS

logger = logging.getLogger(__name__)

# Type alias
Sections = dict[str, str]


def split_sections(text: str) -> Sections:
    """Split a resume into named sections.

    Scans line-by-line for heading-like lines (short, possibly title-cased,
    matching a known section keyword). Groups subsequent lines under the
    matched section label.

    Args:
        text: Plain-text resume content.

    Returns:
        Dict mapping section label (``"experience"``, ``"skills"``,
        ``"education"``, ``"projects"``, ``"summary"``, ``"other"``)
        to the text belonging to that section.
    """
    lines = text.splitlines()
    sections: Sections = {}
    current_label = "other"
    buffer: list[str] = []

    # Build a flat map: keyword → canonical label
    keyword_to_label: dict[str, str] = {}
    for label, keywords in SECTION_HEADERS.items():
        for kw in keywords:
            keyword_to_label[kw.lower()] = label

    for line in lines:
        stripped = line.strip()
        label = _detect_section_header(stripped, keyword_to_label)
        if label is not None:
            # Save previous buffer
            if buffer:
                sections[current_label] = sections.get(current_label, "") + "\n".join(buffer) + "\n"
            current_label = label
            buffer = []
        else:
            buffer.append(line)

    # Flush final buffer
    if buffer:
        sections[current_label] = sections.get(current_label, "") + "\n".join(buffer) + "\n"

    # Strip each section's text
    sections = {k: v.strip() for k, v in sections.items() if v.strip()}

    if not sections:
        sections["other"] = text

    logger.info("Detected sections: %s", list(sections.keys()))
    return sections


def get_section(sections: Sections, label: str) -> str:
    """Retrieve the text for a given section label, or empty string.

    Args:
        sections: Dict returned by :func:`split_sections`.
        label: Section label to retrieve (e.g. ``"experience"``).

    Returns:
        Section text or empty string if the section was not found.
    """
    return sections.get(label, "")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _detect_section_header(line: str, keyword_map: dict[str, str]) -> str | None:
    """Return the section label if *line* looks like a section header.

    A header line is defined as:
    - Short (≤ 60 characters).
    - Matches one of the known section keywords (case-insensitive, full-word).
    - Optionally preceded/followed by common decorators (``:``, ``-``, ``=``).

    Args:
        line: A single stripped line of text.
        keyword_map: Mapping from lowercase keyword to canonical section label.

    Returns:
        Section label string, or ``None`` if the line is not a header.
    """
    if not line or len(line) > 60:
        return None

    # Strip common decorators
    clean = re.sub(r"[:\-=_#*]+", " ", line).strip().lower()

    for keyword, label in keyword_map.items():
        pattern = re.compile(
            r"(?:^|\b)" + re.escape(keyword) + r"(?:\b|$)"
        )
        if pattern.search(clean):
            return label

    return None

