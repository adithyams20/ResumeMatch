"""
parsers/resume_parser.py — Document-to-text parser.

Supports PDF (pdfplumber), DOCX (python-docx), and plain TXT files.
Raises ValueError for unsupported formats or blank/scanned PDFs.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum characters required to consider a document non-empty
_MIN_CONTENT_LENGTH: int = 50


def parse_resume(file_path: str | Path) -> str:
    """Parse a resume file and return its plain-text content.

    Args:
        file_path: Path to the resume file (.pdf, .docx, or .txt).

    Returns:
        Cleaned plain-text string of the resume content.

    Raises:
        ValueError: If the file format is unsupported, the file is empty,
                    or the PDF appears to be a scanned image with no text.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    logger.info("Parsing resume: %s (format: %s)", path.name, suffix)

    if suffix == ".pdf":
        text = _parse_pdf(path)
    elif suffix == ".docx":
        text = _parse_docx(path)
    elif suffix in {".txt", ".text", ".md"}:
        text = _parse_txt(path)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            "Please upload a PDF, DOCX, or TXT file."
        )

    text = _clean_text(text)

    if len(text.strip()) < _MIN_CONTENT_LENGTH:
        raise ValueError(
            "The document appears to be empty or is a scanned image with no "
            "extractable text. Please upload a text-based PDF or DOCX file."
        )

    logger.info("Parsed %d characters from %s", len(text), path.name)
    return text


def parse_resume_bytes(content: bytes, filename: str) -> str:
    """Parse a resume from raw bytes (e.g., from Streamlit's file uploader).

    Args:
        content: Raw file bytes.
        filename: Original filename including extension.

    Returns:
        Cleaned plain-text string of the resume content.

    Raises:
        ValueError: On unsupported format, empty, or scanned documents.
    """
    import tempfile

    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        return parse_resume(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_pdf(path: Path) -> str:
    """Extract text from a PDF using pdfplumber."""
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("pdfplumber is required for PDF parsing.") from exc

    pages_text: list[str] = []
    with pdfplumber.open(path) as pdf:
        if not pdf.pages:
            raise ValueError("The PDF has no pages.")
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)

    return "\n".join(pages_text)


def _parse_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError("python-docx is required for DOCX parsing.") from exc

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def _parse_txt(path: Path) -> str:
    """Read a plain-text file."""
    return path.read_text(encoding="utf-8", errors="replace")


def _clean_text(text: str) -> str:
    """Normalise whitespace and remove null bytes."""
    text = text.replace("\x00", "")
    # Collapse runs of spaces / tabs but preserve newlines
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

